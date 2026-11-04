"""Remove the surfels floating in front of the scene-B mirror from a trained EnvGS run (2026-10-26).

The depth normals of the LiDAR runs show bumps on the mirror: surfels 0.5-5 cm in front of the mirror front face
(the SM_Mirror mesh is ~8.6 mm thick, front face at +2.6 mm from its mid-plane). This writes a copy of latest.npz without
the surfels inside the mirror disc (radius < --rad_frac * r) whose offset from the front face (towards the room) is in
[--dmin, --dmax], to data/trained_model/.../envgs_<run>_<tag>/latest.npz, plus a copy of the exp config, so the usual
eval (eval_r300lp_lidar.sh <run>_<tag>) runs on it. The source run is not touched.

    python prune_mirror_front.py <run> [--dmin 0.005] [--dmax 0.05] [--tag pf05]   (EnvGS repo root)
"""
import argparse
import json
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--dmin', type=float, default=0.005)
ap.add_argument('--dmax', type=float, default=0.05)
ap.add_argument('--rad_frac', type=float, default=0.97)
ap.add_argument('--tag', default='pf05')
ap.add_argument('--pt', action='store_true', help='also write a pruned latest.pt (model + Adam state) to fine-tune from')
ap.add_argument('--keep', action='store_true', help='prune nothing (control run with the same files)')
ap.add_argument('--scene', default='/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300_lamppanel/lidar_scene.npz')
a = ap.parse_args()

sc = np.load(a.scene, allow_pickle=True)
names = json.loads(str(sc['object_names'].item()))
V = sc['tris'][sc['tri_obj'] == names.index('SM_Mirror')].reshape(-1, 3).astype(np.float64)
C = V.mean(0); _, _, vt = np.linalg.svd(V - C); N = vt[2]
N = N if N @ np.array([0.8660254, 0.5, 0.]) > 0 else -N  # towards the room (MirrorBack is on the - side)
U1 = np.array([0., 0., 1.]); U1 -= (U1 @ N) * N; U1 /= np.linalg.norm(U1); U2 = np.cross(N, U1); B = np.stack([U1, U2], 1)
d0 = (V - C) @ N; front = C + d0.max() * N
uv = (V - C) @ B; ctr, r = (uv.min(0) + uv.max(0)) / 2, (uv.max(0) - uv.min(0)).min() / 2

src = f'data/trained_model/envgs/mirror_b/envgs_{a.run}'
dst = f'{src}_{a.tag}'
g = dict(np.load(f'{src}/latest.npz'))
x = g['sampler.pcd._xyz'].astype(np.float64)
op = 1 / (1 + np.exp(-g['sampler.pcd._opacity'][:, 0]))
d = (x - front) @ N
rad = np.linalg.norm((x - C) @ B - ctr, axis=1)
rm = (rad < a.rad_frac * r) & (d >= a.dmin) & (d <= a.dmax)
if a.keep: rm[:] = False
n = len(x)
for k in list(g):
    if k.startswith('sampler.pcd.') and g[k].ndim >= 1 and g[k].shape[0] == n:
        g[k] = g[k][~rm]
os.makedirs(dst, exist_ok=True)
np.savez_compressed(f'{dst}/latest.npz', **g)
if a.pt:
    import torch
    ck = torch.load(f'{src}/latest.pt', 'cpu', weights_only=False)
    keep = torch.as_tensor(~rm)
    for k, v in ck['model'].items():
        if k.startswith('sampler.pcd.') and v.ndim >= 1 and v.shape[0] == n: ck['model'][k] = v[keep]
    for i, pg in enumerate(ck['optimizer']['param_groups']):
        if pg.get('name', '').startswith('sampler.pcd.'):
            for j in pg['params']:
                for kk in ('exp_avg', 'exp_avg_sq'):
                    if j in ck['optimizer']['state'] and kk in ck['optimizer']['state'][j]:
                        assert ck['optimizer']['state'][j][kk].shape[0] == n, (pg['name'], kk)
                        ck['optimizer']['state'][j][kk] = ck['optimizer']['state'][j][kk][keep]
    torch.save(ck, f'{dst}/latest.pt')
cfg = 'configs/exps/envgs/mirror_b/envgs_{}.yaml'
shutil.copy(cfg.format(a.run), cfg.format(f'{a.run}_{a.tag}'))
print(f'{a.run}: removed {rm.sum()} of {n} surfels (sum opacity {op[rm].sum():.0f}, opacity>0.5: {(rm & (op > 0.5)).sum()}), '
      f'front-face offset [{a.dmin * 100:g}, {a.dmax * 100:g}] cm, radius < {a.rad_frac:g} r = {a.rad_frac * r:.3f} m -> {dst}')
