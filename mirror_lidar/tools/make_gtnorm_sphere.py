"""GT normal prior for a spherical mirror (2026-10-30, corridor c480; the curved counterpart of make_mirror_normal_variants.py
_gtnorm): <out>/normals = <src>/normals (StableNormal) with the GT mirror pixels (persp/masks == 3) replaced by the GT sphere
normal where the pixel-centre camera ray hits the sphere, in camera space (OpenCV, n_cam = R_w2c n_world, stored as (n + 1) / 2
like lidar_mirror_normals_curved.py); <out>/mirror_masks = the same GT mirror pixels (255). No rim band: the GT is exact.
images / sparse / intri / extri are symlinked to <src>; envs/points3D.ply is copied (same env init for every run).

    python make_gtnorm_sphere.py <src easyvolcap root> <out> <blender dataset dir> --sphere cx cy cz R [--envs-from <root>]
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('blender_ds')
    ap.add_argument('--sphere', type=float, nargs=4, required=True)
    ap.add_argument('--envs-from', default=None, help='root whose envs/points3D.ply is copied (default <src>)')
    a = ap.parse_args()
    c, R = np.array(a.sphere[:3]), a.sphere[3]
    os.makedirs(a.out, exist_ok=True)
    for f in ('images', 'sparse', 'intri.yml', 'extri.yml'):
        if not os.path.exists(os.path.join(a.out, f)): os.symlink(os.path.realpath(os.path.join(a.src, f)), os.path.join(a.out, f))
    os.makedirs(os.path.join(a.out, 'envs'), exist_ok=True)
    shutil.copy(os.path.join(a.envs_from or a.src, 'envs', 'points3D.ply'), os.path.join(a.out, 'envs', 'points3D.ply'))
    tj = json.load(open(os.path.join(a.blender_ds, 'persp', 'transforms.json')))
    names = sorted(os.listdir(os.path.join(a.src, 'normals')))
    fi, fe = cv2.FileStorage(os.path.join(a.src, 'intri.yml'), 0), cv2.FileStorage(os.path.join(a.src, 'extri.yml'), 0)
    err, npx = [], 0
    for k, n in enumerate(names):
        nm = cv2.imread(os.path.join(a.src, 'normals', n, '000000.png'), -1)
        H, W = nm.shape[:2]; scale = 65535.0 if nm.dtype == np.uint16 else 255.0
        m = cv2.imread(os.path.join(a.blender_ds, 'persp', tj['frames'][k]['mask_path']), -1) == 3
        assert m.shape == (H, W)
        K = fi.getNode(f'K_{n}').mat(); Rw = fe.getNode(f'Rot_{n}').mat(); T = fe.getNode(f'T_{n}').mat()[:, 0]
        o = -Rw.T @ T
        ys, xs = np.nonzero(m)
        d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1) @ Rw  # world dirs
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        oc = o - c; b = d @ oc; disc = b * b - (oc @ oc - R * R)
        hit = disc > 0
        t = -b - np.sqrt(np.maximum(disc, 0))
        nw = (o + d * t[:, None] - c) / R
        ncam = nw @ Rw.T
        res = nm.copy()
        old = nm[ys[hit], xs[hit]][:, ::-1].astype(np.float64) / scale * 2 - 1
        old /= np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-9)
        err.append(np.degrees(np.arccos(np.clip((old * ncam[hit]).sum(1), -1, 1))))
        res[ys[hit], xs[hit]] = np.round((ncam[hit] + 1) / 2 * scale)[:, ::-1].astype(nm.dtype)
        res[ys[~hit], xs[~hit]] = 0
        os.makedirs(os.path.join(a.out, 'normals', n), exist_ok=True)
        cv2.imwrite(os.path.join(a.out, 'normals', n, '000000.png'), res)
        mk = np.zeros((H, W), np.uint8); mk[ys[hit], xs[hit]] = 255
        os.makedirs(os.path.join(a.out, 'mirror_masks', n), exist_ok=True)
        cv2.imwrite(os.path.join(a.out, 'mirror_masks', n, '000000.png'), mk)
        npx += int(hit.sum())
        if (~hit).sum() > 0.01 * len(hit): print(f'{n}: {(~hit).sum()} of {len(hit)} GT mirror px miss the sphere')
    e = np.concatenate(err) if err else np.zeros(1)
    print(f'{len(names)} cameras, {npx} mirror px; StableNormal vs GT sphere on them: median {np.median(e):.1f} deg (expected large)')


if __name__ == '__main__':
    main()
