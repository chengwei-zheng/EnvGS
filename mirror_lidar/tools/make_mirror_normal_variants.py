"""Normal-prior ablations for a Blender mirror dataset already converted for EnvGS (2026-10-16, scene B).

StableNormal predicts the normals of the reflected room inside a mirror. Two variants of <data_root>/normals:
  <data_root>_nmask : GT mirror pixels (persp/masks == 3) stored as black -> skipped with norm_ignore_zero=True
  <data_root>_gtnorm: GT mirror pixels replaced by the GT mirror-plane normal in camera space (OpenCV, as EnvGS
                      compares it: n_cam = R_w2c @ n_world, stored as (n + 1) / 2)
images / sparse / intri / extri are symlinked to <data_root>; envs/points3D.ply is copied so that every run starts
from the same environment-Gaussian init.

    python make_mirror_normal_variants.py <easyvolcap data_root> <blender dataset dir> --mirror-normal 1 0 0

Sanity check printed per run: angle between StableNormal and R @ (0, 0, 1) on floor pixels (should be a few deg)
and between the GT normal and StableNormal on mirror pixels (large: StableNormal sees the reflected room).
Camera k = persp/rgb/rgb_k (blender_gt_to_colmap.py keeps the order).
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np


def read_rot(extri, names):
    fs = cv2.FileStorage(extri, cv2.FILE_STORAGE_READ)
    return {n: fs.getNode(f'Rot_{n}').mat() for n in names}


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root')
    ap.add_argument('blender_ds')
    ap.add_argument('--mirror-normal', type=float, nargs=3, required=True, help='world normal of the mirror front')
    ap.add_argument('--floor-obj', default='SM_Wall.001')
    a = ap.parse_args()
    src = os.path.abspath(a.data_root.rstrip('/'))
    names = sorted(os.listdir(os.path.join(src, 'normals')))
    R = read_rot(os.path.join(src, 'extri.yml'), names)
    tf = json.load(open(os.path.join(a.blender_ds, 'persp', 'transforms.json')))
    floor_id = tf['object_names'].index(a.floor_obj) + 1
    n_world = np.array(a.mirror_normal, dtype=np.float64)
    n_world /= np.linalg.norm(n_world)

    outs = {v: src + '_' + v for v in ('nmask', 'gtnorm')}
    for d in outs.values():
        os.makedirs(os.path.join(d, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(d, 'envs'), exist_ok=True)
        for f in ('images', 'sparse', 'intri.yml', 'extri.yml'):
            if not os.path.lexists(os.path.join(d, f)):
                os.symlink(os.path.join(src, f), os.path.join(d, f))
        shutil.copy2(os.path.join(src, 'envs', 'points3D.ply'), os.path.join(d, 'envs', 'points3D.ply'))

    floor_err, mirror_gap, n_mirror_px = [], [], 0
    for n in names:
        k = int(n)
        nm = cv2.imread(os.path.join(src, 'normals', n, '000000.png'), cv2.IMREAD_UNCHANGED)
        assert nm is not None, n
        msk = cv2.imread(os.path.join(a.blender_ds, 'persp', 'masks', f'mask_{k:04d}.png'), cv2.IMREAD_UNCHANGED)
        obj = cv2.imread(os.path.join(a.blender_ds, 'persp', 'obj_id', f'obj_{k:04d}.png'), cv2.IMREAD_UNCHANGED)
        assert msk.shape == nm.shape[:2], (msk.shape, nm.shape)
        scale = 65535.0 if nm.dtype == np.uint16 else 255.0
        rgb = nm[..., ::-1].astype(np.float64) / scale              # cv2 BGR -> RGB = xyz
        pri = rgb * 2 - 1
        pri /= np.maximum(np.linalg.norm(pri, axis=-1, keepdims=True), 1e-8)
        mir = msk == 3
        n_mirror_px += int(mir.sum())
        fl = obj == floor_id
        if fl.any():
            floor_err.append(np.median(angle_deg(pri[fl], R[n] @ np.array([0, 0, 1.0]))))
        n_cam = R[n] @ n_world
        if mir.any():
            mirror_gap.append(np.median(angle_deg(pri[mir], n_cam)))

        black = nm.copy()
        black[mir] = 0
        gt = nm.copy()
        gt[mir] = np.round((n_cam + 1) / 2 * scale)[::-1].astype(nm.dtype)   # back to BGR
        for v, img in (('nmask', black), ('gtnorm', gt)):
            os.makedirs(os.path.join(outs[v], 'normals', n), exist_ok=True)
            cv2.imwrite(os.path.join(outs[v], 'normals', n, '000000.png'), img)

    print(f'{len(names)} cameras, {n_mirror_px / len(names) / msk.size * 100:.1f}% mirror pixels on average')
    print(f'floor: StableNormal vs R@(0,0,1) median angle per frame, median over frames {np.median(floor_err):.1f} deg '
          f'(max {np.max(floor_err):.1f})  <- validates the camera-frame convention')
    print(f'mirror: StableNormal vs GT plane normal, median over {len(mirror_gap)} frames {np.median(mirror_gap):.1f} deg')
    print('->', ', '.join(outs.values()))


if __name__ == '__main__':
    main()
