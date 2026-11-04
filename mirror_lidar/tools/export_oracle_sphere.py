"""Oracle-normal test for the curved mirror (2026-10-30, corridor c480): export_raw.py with the reflected rays of the mirror
pixels built from the GT sphere instead of the rendered surfel normal / depth. Inference only, the model is unchanged.

Pixels whose camera ray hits the GT sphere cap (half-angle ORACLE_HALF deg about the mirror axis) within ORACLE_TOL of the
rendered depth (i.e. the mirror is the visible surface, not an occluder in front of it) get
  ORACLE_MODE=n   GT sphere normal, ray origin from the rendered depth
  ORACLE_MODE=no  GT sphere normal and the GT sphere hit point as origin
Only the reflected rays change: NORMAL / DEPTH outputs and beta stay the model's own, unless ORACLE_BETA forces beta there
(the c480 runs have beta ~0.02 on the mirror: the reflection is baked into the surfel colours). Same arguments as export_raw.py;
use a new save_tag (results are never overwritten).
    ORACLE_SPHERE=cx,cy,cz,R ORACLE_AXIS=ax,ay,az ORACLE_MODE=n python mirror_lidar/tools/export_oracle_sphere.py -c ... save_tag=oracle_n_raw
"""
import os
import sys
import runpy

# EasyVolcap parses sys.argv when it is first imported (see export_raw.py)
# ORACLE_TASK=gui: open the evc-gui viewer with the same patched reflected rays (gui_oracle_sphere.sh) instead of exporting
sys.argv = ['main.py', '-t', os.environ.get('ORACLE_TASK', 'test')] + sys.argv[1:]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import easyvolcap.runners.visualizers.volumetric_video_visualizer as vvv  # noqa: E402
import easyvolcap.models.samplers.envgs_sampler as es  # noqa: E402

S = [float(v) for v in os.environ['ORACLE_SPHERE'].split(',')]
AX = torch.tensor([float(v) for v in os.environ['ORACLE_AXIS'].split(',')])
AX = AX / AX.norm()
COS = torch.cos(torch.deg2rad(torch.tensor(float(os.environ.get('ORACLE_HALF', '36.0')))))
TOL = float(os.environ.get('ORACLE_TOL', '0.15'))
MODE = os.environ.get('ORACLE_MODE', 'n')
BETA = os.environ.get('ORACLE_BETA')  # if set: beta on the oracle pixels forced to this value (kept for the final blend)
_orig = es.EnvGSSampler.get_reflect_rays
stat = [0, 0]


def get_reflect_rays(self, ray_o, ray_d, coords, output, batch):
    n0, d0 = output.norm_map, output.dpt_map
    c = ray_o.new_tensor(S[:3]); R = S[3]; ax = AX.to(ray_o)
    oc = ray_o - c
    a = (ray_d * ray_d).sum(-1, keepdim=True)
    b = (oc * ray_d).sum(-1, keepdim=True)
    disc = b * b - a * ((oc * oc).sum(-1, keepdim=True) - R * R)
    t = (-b - disc.clamp_min(0).sqrt()) / a                       # near (convex, outside) intersection
    X = ray_o + ray_d * t
    u = (X - c) / R
    m = (disc > 0) & (t > 0) & ((u * ax).sum(-1, keepdim=True) > COS.to(ray_o)) & ((t - d0).abs() < TOL)
    if MODE != 'none': output.norm_map = torch.where(m, u, n0)  # MODE=none: model normals (only beta forced)
    if MODE == 'no': output.dpt_map = torch.where(m, t, d0)
    if BETA is not None: output.spec_map = torch.where(m, torch.full_like(output.spec_map, float(BETA)), output.spec_map)
    stat[0] += int(m.sum()); stat[1] += m.numel()
    try:
        return _orig(self, ray_o, ray_d, coords, output, batch)
    finally:
        output.norm_map, output.dpt_map = n0, d0


def ref_surface(self, o, d, X, N):
    # sub-pixel rays of the multi-sampled reflection (sampler ref_ss > 1): the same GT sphere test per sub-pixel ray
    if MODE == 'none': return X, N
    c = o.new_tensor(S[:3]); R = S[3]; ax = AX.to(o)
    oc = o - c
    a = (d * d).sum(-1, keepdim=True)
    b = (oc * d).sum(-1, keepdim=True)
    disc = b * b - a * ((oc * oc).sum(-1, keepdim=True) - R * R)
    t = (-b - disc.clamp_min(0).sqrt()) / a
    Xg = o + d * t
    u = (Xg - c) / R
    t0 = ((X - o) * d).sum(-1, keepdim=True) / a                  # rendered (interpolated) depth along the same ray
    m = (disc > 0) & (t > 0) & ((u * ax).sum(-1, keepdim=True) > COS.to(o)) & ((t - t0).abs() < TOL)
    return (torch.where(m, Xg, X) if MODE == 'no' else X), torch.where(m, u, N)


es.EnvGSSampler.get_reflect_rays = get_reflect_rays
es.EnvGSSampler.ref_surface = ref_surface
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
    print(f'oracle {MODE}: replaced {stat[0]} / {stat[1]} px ({100 * stat[0] / max(stat[1], 1):.2f}%)', flush=True)
