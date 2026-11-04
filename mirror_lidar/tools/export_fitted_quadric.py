"""Fitted-surface test (2026-11-02, scene B_convex cv300): export_raw.py / evc-gui with the reflected rays of the mirror pixels
built from the LiDAR-fitted global quadric (lidar_mirror_normals_curved.py --surface quadric --export-ply: patches.json)
instead of the rendered surfel normal / depth. Unlike export_fitted_nmap.py (painted per-dataset-camera normal maps) this
intersects every camera ray with the surface, so it also works for free viewpoints in the viewer. Inference only.
(renamed 2026-11-02 from export_oracle_quadric.py: no GT is used, "oracle" stays reserved for GT-based tests.)

Pixels whose camera ray hits the quadric inside the patch extent (disc of the fitted cells' (u, v) footprint) within
FIT_TOL of the rendered depth (the mirror is the visible surface, not an occluder) get
  FIT_MODE=n   fitted normal, ray origin from the rendered depth
  FIT_MODE=no  fitted normal and the fitted hit point as origin (default)
Sensitivity test (simulated LiDAR-camera calibration error): FIT_ROT=deg rotates the fitted surface rigidly about the
patch u axis (horizontal, through the patch centre: the mirror tilts up / down), FIT_ROT_AXIS=u|v|n picks the axis;
FIT_SHIFT=x,y,z (m, world) translates it. NORMAL / DEPTH outputs and beta stay the model's. Same arguments as export_raw.py; FIT_TASK=gui opens the viewer instead.
    FIT_QUADRIC=<dir with patches.json + patch0_cells.ply> python mirror_lidar/tools/export_fitted_quadric.py -c ... save_tag=quadric_raw
"""
import json
import os
import sys
import runpy

# EasyVolcap parses sys.argv when it is first imported (see export_raw.py)
sys.argv = ['main.py', '-t', os.environ.get('FIT_TASK', 'test')] + sys.argv[1:]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from plyfile import PlyData  # noqa: E402

import easyvolcap.runners.visualizers.volumetric_video_visualizer as vvv  # noqa: E402
import easyvolcap.models.samplers.envgs_sampler as es  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_mirror_normals as lmn  # noqa: E402

D = os.environ['FIT_QUADRIC']
TOL = float(os.environ.get('FIT_TOL', '0.15'))
MODE = os.environ.get('FIT_MODE', 'no')
P0 = json.load(open(os.path.join(D, 'patches.json')))[0]
C0 = np.array(P0['center']); RL = lmn.frame_axes(np.array(P0['normal']))      # patch frame, columns u v n
q = np.array(P0['quadric']); M3, S = np.array(P0['fit_scale'][:3]), P0['fit_scale'][3]
A = np.array([[q[0], q[3] / 2, q[4] / 2, q[6] / 2], [q[3] / 2, q[1], q[5] / 2, q[7] / 2],
              [q[4] / 2, q[5] / 2, q[2], q[8] / 2], [q[6] / 2, q[7] / 2, q[8] / 2, q[9]]])   # f(x) = [x 1] A [x 1]^T, x scaled
v = PlyData.read(os.path.join(D, 'patch0_cells.ply'))['vertex']
UV = (np.stack([v['x'], v['y'], v['z']], 1) - C0) @ RL[:, :2]
UV0 = (UV.min(0) + UV.max(0)) / 2; RAD = float(np.linalg.norm(UV - UV0, axis=1).max())   # extent disc
print(f'quadric patch: centre {C0.round(3)}, extent disc radius {RAD:.3f} m, mode {MODE}', flush=True)
ROT = float(os.environ.get('FIT_ROT', '0'))
AXIS = RL[:, 'uvn'.index(os.environ.get('FIT_ROT_AXIS', 'u'))]
SHIFT = np.array([float(x) for x in os.environ.get('FIT_SHIFT', '0,0,0').split(',')])
th = np.radians(ROT); Kx = np.array([[0, -AXIS[2], AXIS[1]], [AXIS[2], 0, -AXIS[0]], [-AXIS[1], AXIS[0], 0]])
RP = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx              # Rodrigues; perturbed surface = RP (x - C0) + C0 + SHIFT
if ROT or SHIFT.any(): print(f'perturbation: rotate {ROT} deg about {os.environ.get("FIT_ROT_AXIS", "u")} {AXIS.round(3)}, shift {SHIFT} m', flush=True)
_orig = es.EnvGSSampler.get_reflect_rays
stat = [0, 0]


def hit(o, d):
    """first intersection of rays (o, d) (..., 3) world with the quadric inside the extent: t, world point, world normal, mask."""
    t_ = lambda x: torch.as_tensor(x, dtype=o.dtype, device=o.device)
    Rl, c0, m3, A_ = t_(RL), t_(C0), t_(M3), t_(A)
    xo = ((o - c0) @ Rl - m3) / S; xd = (d @ Rl) / S                     # scaled patch coordinates, t unchanged
    Xo = torch.cat([xo, torch.ones_like(xo[..., :1])], -1); Xd = torch.cat([xd, torch.zeros_like(xd[..., :1])], -1)
    a = ((Xd @ A_) * Xd).sum(-1); b = 2 * ((Xd @ A_) * Xo).sum(-1); c = ((Xo @ A_) * Xo).sum(-1)
    disc = b * b - 4 * a * c
    sq = disc.clamp_min(0).sqrt()
    lin = a.abs() < 1e-12
    r1 = torch.where(lin, -c / b, (-b - sq) / (2 * a)); r2 = torch.where(lin, -c / b, (-b + sq) / (2 * a))
    lo, hi = torch.minimum(r1, r2), torch.maximum(r1, r2)
    best_t = torch.full_like(a, float('inf')); best_ok = torch.zeros_like(a, dtype=torch.bool)
    for t in (hi, lo):                                                    # nearer valid root wins
        x = xo + t[..., None] * xd
        uv = (x[..., :2] * S + m3[:2]) - t_(UV0)
        z = x[..., 2] * S + m3[2]
        ok = ((disc >= 0) | lin) & (t > 0) & (uv.norm(dim=-1) < RAD) & (z.abs() < 0.3)
        best_t = torch.where(ok, t, best_t); best_ok = best_ok | ok
    x = xo + best_t.nan_to_num(0, 0, 0)[..., None] * xd
    g = (torch.cat([x, torch.ones_like(x[..., :1])], -1) @ A_)[..., :3]
    n = F.normalize(g @ Rl.T, dim=-1)
    n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)
    return best_t, o + d * best_t.nan_to_num(0, 0, 0)[..., None], n, best_ok


def hit_perturbed(o, d):
    """hit() on the rigidly perturbed surface: rays mapped into the unperturbed frame, normals mapped back (t unchanged)."""
    if not (ROT or SHIFT.any()): return hit(o, d)
    Rp, c0, sh = (torch.as_tensor(x, dtype=o.dtype, device=o.device) for x in (RP, C0, SHIFT))
    t, X, n, ok = hit((o - c0 - sh) @ Rp + c0, d @ Rp)                     # Rp^T (o - c0 - sh) + c0 with row vectors
    return t, X @ Rp.T, n @ Rp.T, ok


def get_reflect_rays(self, ray_o, ray_d, coords, output, batch):
    n0, d0 = output.norm_map, output.dpt_map
    dn = ray_d / ray_d.norm(dim=-1, keepdim=True)
    t, X, n, ok = hit_perturbed(ray_o, dn)
    t = t[..., None] / ray_d.norm(dim=-1, keepdim=True)                   # along the unnormalised ray_d (as dpt_map)
    m = ok[..., None] & ((t - d0).abs() < TOL)
    output.norm_map = torch.where(m, n, n0)
    if MODE == 'no': output.dpt_map = torch.where(m, t, d0)
    stat[0] += int(m.sum()); stat[1] += m.numel()
    try:
        return _orig(self, ray_o, ray_d, coords, output, batch)
    finally:
        output.norm_map, output.dpt_map = n0, d0


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
    print(f'fitted quadric {MODE}: replaced {stat[0]} / {stat[1]} px ({100 * stat[0] / max(stat[1], 1):.2f}%)', flush=True)
