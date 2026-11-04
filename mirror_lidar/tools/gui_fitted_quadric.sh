#!/bin/bash
# gui_fitted_quadric.sh <run_tag> [mode n|no] [fit dir]   e.g. cv300_lidarnorm_shared_bfilt_mgsn05   (2026-11-02)
# evc-gui for a scene-B run with the reflected rays of the mirror pixels from the LiDAR-fitted global quadric
# (export_fitted_quadric.py; works from free viewpoints; the model is unchanged, nothing is saved). GPU2 by default.
s=$1
cd /home/yiinqiang/SMBC/EnvGS/EnvGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority} XMODIFIERS=@im=none
export FIT_TASK=gui FIT_MODE=${2:-no} FIT_QUADRIC=${3:-data/result/envgs/mirror_b/mirror_fit_cv300_20261102/quadric_rd06}
CUDA_VISIBLE_DEVICES=${GPU:-2} exec python mirror_lidar/tools/export_fitted_quadric.py -c configs/exps/envgs/mirror_b/envgs_$s.yaml,configs/specs/mirror_b_viewer.yaml exp_name=envgs/mirror_b/envgs_$s < /dev/null
