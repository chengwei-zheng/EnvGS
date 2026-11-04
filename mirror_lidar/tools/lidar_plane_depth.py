"""LiDAR depth with the fitted mirror plane on the mirror (2026-10-26, NEXT_STEPS 10-26).

The first-echo LiDAR depth (lidar_first_echo_depth.py) carries the range noise (synthetic sigma ~2.4 cm). Supervising the
rendered depth with it pulls surfels on the mirror off the plane (bumps 0.5-5 cm in front of the mirror). Here the samples
inside the LiDAR mirror mask (mirror_masks, 255) get the z-depth of the plane fitted to the LiDAR mirror echoes
(lidar_mirror_normals.py, <data_root>_lidarnorm_debug/eval.json patches[0], 'plane' model) along the ray through the
pixel centre; samples in the boundary band (128) are dropped; all other samples are kept unchanged.
--drop_mirror (2026-10-27): drop the mirror samples (255 and 128) instead, i.e. no LiDAR depth on the mirror at all.

    python lidar_plane_depth.py data/datasets/mirror_b/r300lp_lidarnorm data/datasets/mirror_b/r300lp_lidar_depth \
        data/datasets/mirror_b/r300lp_lidar_depth_mplane [--eval <blender_ds>]
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('data_root')
ap.add_argument('depth_dir')
ap.add_argument('out_dir')
ap.add_argument('--drop_mirror', action='store_true', help='drop the mirror samples instead of using the plane depth')
ap.add_argument('--eval', default='', help='blender dataset dir: compare with persp/depth (GT, evaluation only)')
a = ap.parse_args()

P = json.load(open(a.data_root + '_debug/eval.json'))['patches']
assert len(P) == 1 and P[0]['model'] == 'plane', P
pc, pn = np.array(P[0]['center']), np.array(P[0]['normal'])
os.makedirs(a.out_dir, exist_ok=True)
names = sorted(os.listdir(os.path.join(a.data_root, 'images')))
fe, fi = cv2.FileStorage(os.path.join(a.data_root, 'extri.yml'), 0), cv2.FileStorage(os.path.join(a.data_root, 'intri.yml'), 0)
H0, W0 = cv2.imread(os.path.join(a.data_root, 'images', names[0], '000000.png')).shape[:2]
st = dict(n=0, mir=0, band=0, dz=[], e0=[], e1=[])
for n in names:
    d = np.load(os.path.join(a.depth_dir, f'{n}.npz'))
    idx, z, H, W = d['idx'], d['z'].copy(), int(d['H']), int(d['W'])
    R, t, K = fe.getNode(f'Rot_{n}').mat(), fe.getNode(f'T_{n}').mat().ravel(), fi.getNode(f'K_{n}').mat().copy()
    K[0] *= W / W0; K[1] *= H / H0
    m = cv2.imread(os.path.join(a.data_root, 'mirror_masks', n, '000000.png'), -1)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).ravel()[idx]
    py, px = idx // W, idx % W
    dc = np.stack([(px + 0.5 - K[0, 2]) / K[0, 0], (py + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(idx))], 1)  # z = 1
    o = -R.T @ t
    zp = ((pc - o) @ pn) / ((dc @ R) @ pn)                                    # dc @ R = (R^T dc)^T
    mir, band = m == 255, m == 128
    st['dz'].append(z[mir] - zp[mir])
    if a.eval:
        k = int(n)
        import OpenEXR, Imath
        tj = json.load(open(os.path.join(a.eval, 'persp/transforms.json'))); fr = tj['frames'][k]
        g = np.frombuffer(OpenEXR.InputFile(os.path.join(a.eval, 'persp', fr['depth_file_path'])).channel('depth.V', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(tj['h'], tj['w'])
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST).ravel()[idx]
        st['e0'].append(np.abs(z[mir] - g[mir])); st['e1'].append(np.abs(zp[mir] - g[mir]))
    z[mir] = zp[mir]
    keep = ~band & ~mir if a.drop_mirror else ~band
    np.savez(os.path.join(a.out_dir, f'{n}.npz'), idx=idx[keep], z=z[keep].astype(np.float32), H=H, W=W)
    st['n'] += len(idx); st['mir'] += mir.sum(); st['band'] += band.sum()
dz = np.concatenate(st['dz'])
print(f'{len(names)} cameras -> {a.out_dir}: {st["n"]} samples, {st["mir"]} on the mirror {"dropped" if a.drop_mirror else "replaced by the plane depth"}, '
      f'{st["band"]} in the boundary band dropped; LiDAR - plane on the mirror: median {100 * np.median(dz):+.2f} cm, '
      f'MAD-sigma {100 * 1.4826 * np.median(np.abs(dz - np.median(dz))):.2f} cm')
if st['e0']:
    print(f'vs GT depth on the mirror samples (eval only): LiDAR median |err| {100 * np.median(np.concatenate(st["e0"])):.2f} cm, '
          f'plane {100 * np.median(np.concatenate(st["e1"])):.3f} cm')
