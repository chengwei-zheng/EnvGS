#!/bin/bash
# gui_mirror_a.sh <run_tag>   e.g. a220
# Opens the EnvGS evc-gui for configs/exps/envgs/mirror_a/envgs_<run_tag>.yaml on the display GPU (GPU2),
# with the scene-A viewer defaults from configs/specs/mirror_a_viewer.yaml.
s=$1
cd /home/yiinqiang/SMBC/EnvGS/EnvGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
# Keep IBus (XIM) out of the GLFW window: otherwise WASD typed in the viewer is buffered by the IME as pinyin
# and committed later into whatever window gets focus next.
export XMODIFIERS=@im=none
CUDA_VISIBLE_DEVICES=${GPU:-2} exec evc-gui -c configs/exps/envgs/mirror_a/envgs_$s.yaml,configs/specs/mirror_a_viewer.yaml exp_name=envgs/mirror_a/envgs_$s < /dev/null
