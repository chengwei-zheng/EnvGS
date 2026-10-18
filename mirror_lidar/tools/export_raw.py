"""Run EasyVolcap `-t test` but store visualizations that would be written as .exr as raw float .npy instead.

The envgs env ships OpenCV 5.0 without an EXR writer, so EasyVolcap's save_image silently fails on .exr.
Usage (from the EnvGS repo root, envgs env):
    python mirror_lidar/tools/export_raw.py -c <exp.yaml> exp_name=<exp> runner_cfg.visualizer_cfg.save_tag=rawdepth \
        runner_cfg.visualizer_cfg.vis_ext=.exr runner_cfg.visualizer_cfg.dpt_curve=linear \
        runner_cfg.visualizer_cfg.store_alpha_channel=False runner_cfg.visualizer_cfg.store_ground_truth=False \
        runner_cfg.visualizer_cfg.store_image_error=False 'runner_cfg.visualizer_cfg.types=[RENDER,DEPTH]'
RENDER -> <...>/RENDER/frameXXXX_cameraYYYY.npy (H, W, 3) float RGB, DEPTH -> (H, W, 1) float z-depth (COLMAP units).
"""
import os
import sys
import runpy

# EasyVolcap parses sys.argv when it is first imported, so the test-mode argv must be set before that
sys.argv = ['main.py', '-t', 'test'] + sys.argv[1:]

import numpy as np
import torch

import easyvolcap.runners.visualizers.volumetric_video_visualizer as vvv

_orig_save_image = vvv.save_image


def save_image(img_path, img, *args, **kwargs):
    if img_path.endswith('.exr'):
        if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        np.save(img_path[:-4] + '.npy', np.asarray(img, dtype=np.float32))
        return True
    return _orig_save_image(img_path, img, *args, **kwargs)


vvv.save_image = save_image
runpy.run_path(os.path.join(os.getcwd(), 'easyvolcap/scripts/main.py'), run_name='__main__')
