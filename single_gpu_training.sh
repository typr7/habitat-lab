#!/usr/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=0,1,2
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

set -x

python -u -m habitat_baselines.run \
    --config-name=objectnav/ver_objectnav.no_pretrained.procthor-hab_414720000.yaml