#!/bin/bash
# gui_bathroom.sh <run_tag>   e.g. walk400_2mirrors, walk400_2mirrors_spec01, walk400v2
# Opens the EnvGS evc-gui for configs/exps/envgs/bathroom/envgs_<run_tag>.yaml on the display GPU (GPU2),
# with the bathroom viewer defaults from configs/specs/bathroom_viewer.yaml.
s=$1
cd /home/yiinqiang/SMBC/EnvGS/EnvGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
CUDA_VISIBLE_DEVICES=${GPU:-2} exec evc-gui -c configs/exps/envgs/bathroom/envgs_$s.yaml,configs/specs/bathroom_viewer.yaml exp_name=envgs/bathroom/envgs_$s < /dev/null
