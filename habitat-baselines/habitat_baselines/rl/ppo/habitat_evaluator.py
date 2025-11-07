import os
from collections import defaultdict
from typing import Any, Dict, List
from PIL import Image
from pathlib import Path

import numpy as np
import torch
import tqdm
import json
import copy

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    maps
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info


def create_nav_id(scene_id: str, episode_id: str) -> str:
    scene_id: str = scene_id.split('/')[-1]
    return f'scene_id={scene_id}.episode_id={episode_id}'

def create_nav_data_json(nav_data: dict, config) -> dict:
    action_mapping = list(config.habitat.task.actions.keys())

    action_sequence = nav_data['action_sequence']
    action_sequence = [action_mapping[action] for action in action_sequence]

    failure_cause = None
    if nav_data['success'] != 1.0:
        if nav_data['action_sequence'][-1] == 0:
            failure_cause = 'stop too far'
        else:
            failure_cause = 'timeout'

    return {
        'metric': {
            'success': nav_data['success'],
            'spl': nav_data['spl'],
            'soft_spl': nav_data['soft_spl'],
            'distance_to_goal': nav_data['distance_to_goal']
        },
        'navigation': {
            'failure_cause': failure_cause,
            'goal_category': nav_data['goal_category'],
            'action_sequence': action_sequence,
            'trajectory': nav_data['trajectory']
        },
        'visualization': {
            'video_name': nav_data.get('video_name', None),
            'top_down_video_name': nav_data.get('top_down_video_name', None)
        }
    }

class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        if config.habitat_baselines.eval.collect_nav_data:
            # trajectory
            action_seq = [list() for _ in range(envs.num_envs)]
            pose_seq = [list() for _ in range(envs.num_envs)]
            top_down_video_dir_path = Path('nav_data/top_down_video')
            top_down_video_dir_path = str(top_down_video_dir_path.resolve())

            # nav data json
            json_dir_path = Path('nav_data/json')
            json_dir_path = str(json_dir_path.resolve())
            nav_data = defaultdict(lambda: dict())

            dc_metric_list = ['agent_pose']

            os.makedirs(json_dir_path, exist_ok=True)
            os.makedirs(top_down_video_dir_path, exist_ok=True)
        else:
            action_seq = None
            pose_seq = None
            top_down_video_dir_path = None
            json_dir_path = None
            nav_data = None
            dc_metric_list = []

        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]

            if config.habitat_baselines.eval.collect_nav_data:
                top_down_maps = [[] for _ in range(envs.num_envs)]
            else:
                top_down_maps = None
        else:
            rgb_frames = None
            top_down_maps = None
        
        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            current_episodes_goal_category = envs.current_episodes_goal_category()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )
            
            if config.habitat_baselines.eval.collect_nav_data:
                for i, action in enumerate(action_data.actions):
                    action_seq[i].append(action[0].item())

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            if config.habitat_baselines.eval.collect_nav_data:
                for i in range(envs.num_envs):
                    info = infos[i]
                    pose_seq[i].append(info['agent_pose'] + (info['distance_to_goal'],))

                    for key in dc_metric_list:
                        info.pop(key, None)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if (k not in rank0_keys) and (k not in dc_metric_list) and (k != 'top_down_map')
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)
                    
                    if config.habitat_baselines.eval.collect_nav_data:
                        top_down_map = maps.colorize_draw_agent_and_fit_to_height(infos[i]['top_down_map'], 512)
                        top_down_maps[i].append(top_down_map)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # save the final frame and the first frame of next episode
                    if config.habitat_baselines.eval.collect_nav_data:
                        # store episode metrics
                        nav_data[k]['success'] = disp_info['success']
                        nav_data[k]['spl'] = disp_info['spl']
                        nav_data[k]['soft_spl'] = disp_info['soft_spl']
                        nav_data[k]['distance_to_goal'] = disp_info['distance_to_goal']

                        # store goal category
                        nav_data[k]['goal_category'] = current_episodes_goal_category[i]

                        # store agent action sequence
                        nav_data[k]['action_sequence'] = copy.deepcopy(action_seq[i])
                        action_seq[i].clear()

                        # store agent trajectory
                        nav_data[k]['trajectory'] = copy.deepcopy(pose_seq[i])
                        pose_seq[i].clear()

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        success = disp_info['success']
                        if success != 1.0:
                            video_name = generate_video(
                                video_option=config.habitat_baselines.eval.video_option,
                                video_dir=config.habitat_baselines.video_dir,
                                # Since the final frame is the start frame of the next episode.
                                images=rgb_frames[i][:-1],
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(disp_info),
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                            )

                            if config.habitat_baselines.eval.collect_nav_data:
                                nav_data[k]['video_name'] = video_name

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                        if config.habitat_baselines.eval.collect_nav_data:
                            if success != 1.0:
                                video_name = generate_video(
                                    video_option=config.habitat_baselines.eval.video_option,
                                    video_dir=top_down_video_dir_path,
                                    # Since the final frame is the start frame of the next episode.
                                    images=top_down_maps[i],
                                    episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                                    checkpoint_idx=checkpoint_index,
                                    metrics=extract_scalars_from_info(disp_info),
                                    fps=config.habitat_baselines.video_fps,
                                    tb_writer=writer,
                                    keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                                )

                                nav_data[k]['top_down_video_name'] = video_name

                            top_down_maps[i].clear()
                    
                    if config.habitat_baselines.eval.collect_nav_data:
                        episode_nav_data = nav_data.pop(k)
                        episode_nav_data = create_nav_data_json(episode_nav_data, config)
                        with open(
                            os.path.join(json_dir_path, create_nav_id(k[0], k[1]) + '.json'),
                            'w'
                        ) as fp:
                            json.dump(episode_nav_data, fp, indent=4)

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                top_down_maps,
                action_seq,
                pose_seq
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                top_down_maps,
                action_seq,
                pose_seq
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
