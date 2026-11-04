#!/bin/bash
# eval_mirror_b_run.sh <exp_tag> <dataset_dir_name> <gpu>   e.g. eval_mirror_b_run.sh b300_nmask dataset_b300 1
# Export raw RGB/depth of a finished EnvGS mirror_b run, convert to npz, run the GT eval (GT poses, scale 1).
s=$1; d=$2; g=$3
E=/home/yiinqiang/SMBC/EnvGS/EnvGS; B=${B:-/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b}  # r300: B=<proj_indoor_mirror_b_rot30>
C=${C:-/home/yiinqiang/SMBC/EnvGS/data/datasets/original/mirror_b/${d#dataset_}}  # r300lp: C=<original/mirror_b/r300lp>
R=$E/data/result/envgs/mirror_b/envgs_$s
cd $E
bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs; CUDA_VISIBLE_DEVICES=$g python mirror_lidar/tools/export_raw.py -c configs/exps/envgs/mirror_b/envgs_$s.yaml exp_name=envgs/mirror_b/envgs_$s runner_cfg.visualizer_cfg.save_tag=rawdepth runner_cfg.visualizer_cfg.vis_ext=.exr runner_cfg.visualizer_cfg.dpt_curve=linear runner_cfg.visualizer_cfg.store_alpha_channel=False runner_cfg.visualizer_cfg.store_ground_truth=False runner_cfg.visualizer_cfg.store_image_error=False 'runner_cfg.visualizer_cfg.types=[RENDER,DEPTH]' < /dev/null > data/logs/export_mirror_b_$s.log 2>&1"
~/miniconda3/envs/envgs/bin/python mirror_lidar/tools/envgs_to_eval_npz.py --export_dir $R/rawdepth --images_dir $C/images --out_dir $R/eval_npz
cd /home/yiinqiang/SMBC/gsplat/gsplat/script
CUDA_VISIBLE_DEVICES=$g ~/miniconda3/envs/gsplat/bin/python /home/yiinqiang/SMBC/EnvGS/EnvGS/mirror_lidar/tools/eval_glass_anyname.py --pred_dir $R/eval_npz --pred_depth_key depth --data_dir $C --gt_dir $B/$d/persp --out_dir $R/eval_gt --downscale 2 --split val > $R/eval_gt.log 2>&1
echo "## $s"; sed -n '/=== val/,/Wrote/p' $R/eval_gt.log | grep -v Wrote
