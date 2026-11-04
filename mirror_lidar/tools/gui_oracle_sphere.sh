#!/bin/bash
# gui_oracle_sphere.sh <run_tag> [mode]   e.g. cv300_gtnorm_shared_bfilt_mgsn05 no   (2026-11-02)
# evc-gui for a scene B_convex run with the reflected rays of the mirror pixels built from the GT sphere
# (export_oracle_sphere.py, ORACLE_MODE n|no|none; the model is unchanged, nothing is saved). GPU2 by default, as gui_mirror_b.sh.
s=$1
cd /home/yiinqiang/SMBC/EnvGS/EnvGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority} XMODIFIERS=@im=none
export ORACLE_TASK=gui ORACLE_MODE=${2:-no} ORACLE_HALF=33 ORACLE_SPHERE=0.57234,-0.17039,1.30028,1.0 ORACLE_AXIS=0.8660,0.5,0
CUDA_VISIBLE_DEVICES=${GPU:-2} exec python mirror_lidar/tools/export_oracle_sphere.py -c configs/exps/envgs/mirror_b/envgs_$s.yaml,configs/specs/mirror_b_viewer.yaml exp_name=envgs/mirror_b/envgs_$s < /dev/null
