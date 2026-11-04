#!/bin/bash
# gui_fitted_nmap.sh <run_tag> [nmap root tag]   e.g. cv300_lidarnorm_shared_bfilt_mgsn05 cv300_lidarnorm_quadric   (2026-11-02)
# evc-gui for a scene-B run with the reflected rays of the mirror pixels built from a painted normal map
# (export_fitted_nmap.py: data/datasets/mirror_b/<nmap root>/normals on mirror_masks == 255; the model is unchanged, nothing is saved).
s=$1
cd /home/yiinqiang/SMBC/EnvGS/EnvGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority} XMODIFIERS=@im=none
export FIT_TASK=gui FIT_NMAP_ROOT=data/datasets/mirror_b/${2:-cv300_lidarnorm_quadric}
CUDA_VISIBLE_DEVICES=${GPU:-2} exec python mirror_lidar/tools/export_fitted_nmap.py -c configs/exps/envgs/mirror_b/envgs_$s.yaml,configs/specs/mirror_b_viewer.yaml exp_name=envgs/mirror_b/envgs_$s < /dev/null
