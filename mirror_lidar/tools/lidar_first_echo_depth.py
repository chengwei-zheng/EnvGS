"""Sparse first-echo LiDAR depth per camera, for LidarDepthSupervisor (2026-10-20, mirror_lidar/LIDAR_MIRROR_NORMAL.md).

For every camera, the nearest (first, by RANGE) echo of every zone of the same-index LiDAR frame
(th_s02_format/extracted/pointclouds_multiecho, ranges + xyz only, no echo types) is moved to world coordinates with the
sensor pose (gt_poses_tum.txt), projected into the camera (EasyVolcap intri/extri, scaled to the training resolution)
and stored as camera z-depth in <out_dir>/<camera name>.npz: idx (flat pixel index), z, H, W. Nearest sample wins in a pixel.
Real data: the LiDAR frame nearest in time (+ pose interpolation) would replace "same index".

    python lidar_first_echo_depth.py data/datasets/mirror_b/r300 ~/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300 \
        data/datasets/mirror_b/r300_lidar_depth --ratio 0.5 [--eval]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_mirror_normals import load_lidar, nearest_echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root')
    ap.add_argument('blender_ds')
    ap.add_argument('out_dir')
    ap.add_argument('--ratio', type=float, default=0.5, help='training resolution / image resolution (dataset ratio)')
    ap.add_argument('--eval', action='store_true', help='compare with persp/depth (GT, evaluation only)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = sorted(os.listdir(os.path.join(a.data_root, 'images')))
    fe, fi = cv2.FileStorage(os.path.join(a.data_root, 'extri.yml'), 0), cv2.FileStorage(os.path.join(a.data_root, 'intri.yml'), 0)
    files, T = load_lidar(a.blender_ds)
    assert len(files) == len(names)
    H0, W0 = cv2.imread(os.path.join(a.data_root, 'images', names[0], '000000.png')).shape[:2]
    H, W = int(H0 * a.ratio), int(W0 * a.ratio)
    cov, err, n_s = [], [], []
    for n in names:
        k = int(n)
        R, t, K = fe.getNode(f'Rot_{n}').mat(), fe.getNode(f'T_{n}').mat().ravel(), fi.getNode(f'K_{n}').mat().copy()
        K[0] *= W / W0; K[1] *= H / H0                     # same intrinsics scaling as the dataset loader
        cnt, _, p1 = nearest_echo(files[k])
        P = p1[cnt >= 1] @ T[k][:3, :3].T + T[k][:3, 3]
        xc = P @ R.T + t
        z = xc[:, 2]
        u, v = xc[:, 0] / z * K[0, 0] + K[0, 2], xc[:, 1] / z * K[1, 1] + K[1, 2]
        m = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = v[m].astype(np.int64) * W + u[m].astype(np.int64)
        o = np.lexsort((z[m], idx))                        # per pixel keep the nearest sample
        idx, zz = idx[o], z[m][o]
        first = np.r_[True, idx[1:] != idx[:-1]]
        idx, zz = idx[first], zz[first].astype(np.float32)
        np.savez(os.path.join(a.out_dir, f'{n}.npz'), idx=idx, z=zz, H=H, W=W)
        cov.append(len(idx) / (H * W)); n_s.append(len(idx))
        if a.eval:
            from eval_synthetic_glass import read_exr_depth
            g = cv2.resize(read_exr_depth(os.path.join(a.blender_ds, 'persp', 'depth', f'depth_{k:04d}.exr'), H0, W0), (W, H),
                           interpolation=cv2.INTER_NEAREST).ravel()
            err.append(np.median(np.abs(zz - g[idx])))
    print(f'{len(names)} cameras -> {a.out_dir}: {np.mean(n_s):.0f} LiDAR samples per camera ({100 * np.mean(cov):.1f}% of {W}x{H} pixels)')
    if err: print(f'first-echo z-depth vs GT depth (eval only): median |error| per frame, median over frames {100 * np.median(err):.2f} cm')


if __name__ == '__main__':
    main()
