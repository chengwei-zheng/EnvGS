"""GT-sphere mirror patch for the msurf oracle test (2026-11-04, c480): same npz format as lidar_mirror_normals_curved.py --export-ply
patch<i>.npz (input of samplers/mirror_surface.py), but the quadric is the exact GT sphere and the extent is the full GT cap
(radial distance from the mirror axis < aperture) instead of the LiDAR 1 cm occupancy (c480: the top ~4 rows had no LiDAR echoes).
The patch frame (c, R) and the fit_scale normalisation are copied from the LiDAR fit. Use with type: quadric, learnable: False.
    python make_oracle_sphere_patch.py <lidar patch0.npz> <sphere.json (centre, R, axis, aperture)> <out.npz>
"""
import json
import sys

import numpy as np
from scipy import ndimage

a = dict(np.load(sys.argv[1], allow_pickle=True)); S = json.load(open(sys.argv[2]))
Cw, Rs, AX, AP = np.array(S['centre']), S['R'], np.array(S['axis']), S['aperture']
c, R, cell = a['c'], a['R'], float(a['cell'])
m, s = a['fit_scale'][:3], float(a['fit_scale'][3])
cl = (Cw - c) @ R                                                   # sphere centre in the patch frame
# implicit sphere in normalised coordinates xs = (x - m) / s: s^2 |xs|^2 + 2 s xs.(m - cl) + |m - cl|^2 - Rs^2
d = m - cl
q = np.array([s * s, s * s, s * s, 0, 0, 0, 2 * s * d[0], 2 * s * d[1], 2 * s * d[2], d @ d - Rs * Rs])
q /= np.linalg.norm(q)
# full cap on a 1 cm grid of the patch (u, v) plane, heights = the cap's z in the patch frame
g = np.linspace(-AP, AP, 1201); U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U); V = np.cross(U, AX)
uu, vv = np.meshgrid(g, g); inn = uu ** 2 + vv ** 2 < AP ** 2
X = Cw + uu[inn][:, None] * U + vv[inn][:, None] * V + np.sqrt(Rs * Rs - uu[inn] ** 2 - vv[inn] ** 2)[:, None] * AX
xl = (X - c) @ R
g0 = np.floor(xl[:, :2].min(0) / cell) * cell - 2 * cell
W, H = (np.ceil((xl[:, :2].max(0) - g0) / cell) + 2).astype(int)
jj, ii = np.mgrid[0:H, 0:W]; xy = g0 + (np.stack([ii, jj], -1) + 0.5) * cell
# cap height at each cell centre: intersect the patch-normal line with the sphere (outer root, towards +z = camera side as in the fit)
dz = Rs * Rs - ((xy - cl[:2]) ** 2).sum(-1)
z = np.where(dz > 0, cl[2] + np.sign((Cw + Rs * AX - c) @ R[:, 2] - cl[2]) * np.sqrt(np.maximum(dz, 0)), np.nan)
P = np.concatenate([xy, z[..., None]], -1) @ R.T + c                 # world
rad = np.linalg.norm((P - Cw) - ((P - Cw) @ AX)[..., None] * AX, axis=-1)
ext = (dz > 0) & (rad < AP) & (((P - Cw) @ AX) > 0)
n = (np.nan_to_num(P) - Cw) / Rs @ R                                 # local normal
Hu, Hv = -n[..., 0] / n[..., 2], -n[..., 1] / n[..., 2]
rd = ndimage.distance_transform_edt(ext) * cell
a.update(quadric=q, g0=g0, extent=ext.astype(np.uint8), Hh=np.nan_to_num(z), Hu=np.nan_to_num(Hu), Hv=np.nan_to_num(Hv),
         cells=P[ext], cells_rim_dist=rd[ext], cells_w=np.ones(int(ext.sum())), model=np.array('oracle_gt_sphere'))
np.savez(sys.argv[3], **a)
old = a  # report
print(f'oracle patch: extent {int(ext.sum())} cells ({H} x {W}), LiDAR extent {int(np.load(sys.argv[1])["extent"].sum())}; quadric {np.round(q, 4)} -> {sys.argv[3]}')
