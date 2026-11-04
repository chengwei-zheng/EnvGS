"""Fitted-normal test (2026-11-02, scene B_convex cv300): export_raw.py with the reflected rays of the mirror pixels built from
a painted normal map (e.g. the LiDAR-fitted curved-mirror normals of lidar_mirror_normals_curved.py) instead of the rendered
surfel normal. Inference only, the model is unchanged; the export_oracle_sphere.py counterpart without GT
(renamed 2026-11-02 from export_oracle_nmap.py: no GT is used, "oracle" stays reserved for GT-based tests).

Pixels with <FIT_NMAP_ROOT>/mirror_masks/<cam>/000000.png == 255 (interior; the 128 rim band and the rest keep the model normal)
get the normal of <FIT_NMAP_ROOT>/normals/<cam>/000000.png (camera space OpenCV, stored as (n + 1) / 2 in BGR order),
rotated to world with the batch w2c R. The ray origin stays the rendered depth; NORMAL / DEPTH outputs and beta stay the model's.
Same arguments as export_raw.py; use a new save_tag (results are never overwritten). FIT_TASK=gui opens the viewer instead.
    FIT_NMAP_ROOT=data/datasets/mirror_b/cv300_lidarnorm python mirror_lidar/tools/export_fitted_nmap.py -c ... save_tag=nmap_raw
"""
import os
import sys
import runpy

# EasyVolcap parses sys.argv when it is first imported (see export_raw.py)
sys.argv = ['main.py', '-t', os.environ.get('FIT_TASK', 'test')] + sys.argv[1:]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import easyvolcap.runners.visualizers.volumetric_video_visualizer as vvv  # noqa: E402
import easyvolcap.models.samplers.envgs_sampler as es  # noqa: E402

ROOT = os.environ['FIT_NMAP_ROOT']
_orig = es.EnvGSSampler.get_reflect_rays
_cache = {}
stat = [0, 0]


def load(cam, H, W, device):
    if cam not in _cache:
        n = cv2.imread(os.path.join(ROOT, 'normals', f'{cam:04d}', '000000.png'), -1)
        m = cv2.imread(os.path.join(ROOT, 'mirror_masks', f'{cam:04d}', '000000.png'), -1)
        n = n[..., ::-1].astype(np.float32) / (65535.0 if n.dtype == np.uint16 else 255.0) * 2 - 1
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) == 255
        _cache.clear()
        _cache[cam] = torch.as_tensor(n, device=device).reshape(1, H * W, 3), torch.as_tensor(m, device=device).reshape(1, H * W, 1)
    return _cache[cam]


def get_reflect_rays(self, ray_o, ray_d, coords, output, batch):
    n0 = output.norm_map
    H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
    ncam, m = load(int(batch.meta.camera_index[0]), H, W, ray_o.device)
    R = batch.R[0].to(ray_o)                                           # w2c
    nw = F.normalize(ncam @ R, dim=-1)                                 # n_world = R^T n_cam (row vectors)
    nw = torch.where((nw * ray_d).sum(-1, keepdim=True) > 0, -nw, nw)  # face the camera
    m = m & (n0.norm(dim=-1, keepdim=True) > 0)
    output.norm_map = torch.where(m, nw, n0)
    stat[0] += int(m.sum()); stat[1] += m.numel()
    if os.environ.get('FIT_DEBUG') and m.any():
        cos = (F.normalize(n0, dim=-1) * nw).sum(-1, keepdim=True)[m]
        print(f'cam {int(batch.meta.camera_index[0])} R00 {R[0, 0].item():.3f} model vs map normal: med {torch.rad2deg(torch.acos(cos.clamp(-1, 1))).median().item():.2f} deg', flush=True)
    try:
        return _orig(self, ray_o, ray_d, coords, output, batch)
    finally:
        output.norm_map = n0


es.EnvGSSampler.get_reflect_rays = get_reflect_rays
_orig_save_image = vvv.save_image


def save_image(img_path, img, *args, **kwargs):  # as export_raw.py: .exr -> raw float .npy
    if img_path.endswith('.exr'):
        if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        np.save(img_path[:-4] + '.npy', np.asarray(img, dtype=np.float32))
        return True
    return _orig_save_image(img_path, img, *args, **kwargs)


vvv.save_image = save_image
try:
    runpy.run_path(os.path.join(os.getcwd(), 'easyvolcap/scripts/main.py'), run_name='__main__')
finally:
    print(f'fitted nmap: replaced {stat[0]} / {stat[1]} px ({100 * stat[0] / max(stat[1], 1):.2f}%)', flush=True)
