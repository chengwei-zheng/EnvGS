"""Initial point cloud for the base surfels from the LiDAR first echoes (2026-10-23, mirror_lidar/NEXT_STEPS.md #3).

Replaces the COLMAP triangulation (sparse/0/points3D.ply), which on r300 has 8.7% mirror-virtual points (triangulated
through the mirror). For every TRAINING camera k (val = every 8th, as the dataset split), the nearest-by-range echo of
every zone of the same-index LiDAR frame (ranges + xyz only, no echo types, as lidar_first_echo_depth.py) is moved to world
coordinates with the sensor pose, projected into camera k at full resolution and coloured from that image; points outside
the image or behind other LiDAR points of the same frame (block z-buffer) get no colour and are dropped. All coloured
points are then reduced to one point per voxel: colour = voxel mean, position = centroid of the 3x3x3 voxel block.
Output: a COLMAP-style ply (x, y, z, red, green, blue, as load_sfm_ply / save_sfm_ply) at <out_root>/sparse/0/points3D.ply,
<out_root> = copy of <data_root> made of symlinks except sparse/ (and envs/, regenerated at the first training start).

    python lidar_init_points.py data/datasets/mirror_b/r300lp_lidarnorm ~/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300_lamppanel \
        data/datasets/mirror_b/r300lp_lidarnorm_linit [--vox 0.02] [--gt-surface <ply>]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_mirror_normals import load_lidar, nearest_echo


def save_ply(path, xyz, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(xyz)
    v = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = xyz.T
    v['red'], v['green'], v['blue'] = rgb.T
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n'
                 f'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root', help='EasyVolcap dataset (images/, intri.yml, extri.yml)')
    ap.add_argument('blender_ds')
    ap.add_argument('out_root')
    ap.add_argument('--vox', type=float, default=0.02, help='voxel size (m) for averaging')
    ap.add_argument('--block', type=int, default=8, help='z-buffer block (px, full resolution)')
    ap.add_argument('--zband', type=float, default=0.05, help='keep points within this of the block minimum depth (m)')
    ap.add_argument('--gt-surface', default='', help='ply of points sampled on the GT mesh (evaluation only)')
    ap.add_argument('--mirror', type=float, nargs=6, default=None, help='GT mirror centre + normal (evaluation only)')
    a = ap.parse_args()
    names = sorted(os.listdir(os.path.join(a.data_root, 'images')))
    fe, fi = cv2.FileStorage(os.path.join(a.data_root, 'extri.yml'), 0), cv2.FileStorage(os.path.join(a.data_root, 'intri.yml'), 0)
    files, T = load_lidar(a.blender_ds)
    assert len(files) == len(names)
    Ps, Cs, n_all = [], [], 0
    for n in names:
        k = int(n)
        if k % 8 == 0: continue                              # validation camera
        img = cv2.imread(os.path.join(a.data_root, 'images', n, '000000.png'))[..., ::-1]
        H, W = img.shape[:2]
        R, t, K = fe.getNode(f'Rot_{n}').mat(), fe.getNode(f'T_{n}').mat().ravel(), fi.getNode(f'K_{n}').mat()
        cnt, _, p1 = nearest_echo(files[k])
        P = p1[cnt >= 1] @ T[k][:3, :3].T + T[k][:3, 3]
        n_all += len(P)
        xc = P @ R.T + t
        z = xc[:, 2]
        u, v = xc[:, 0] / z * K[0, 0] + K[0, 2], xc[:, 1] / z * K[1, 1] + K[1, 2]
        m = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        P, z, u, v = P[m], z[m], u[m].astype(np.int64), v[m].astype(np.int64)
        blk = (v // a.block) * (W // a.block + 1) + u // a.block
        zmin = np.full(blk.max() + 1, np.inf); np.minimum.at(zmin, blk, z)
        keep = z < zmin[blk] + a.zband                       # drop points hidden behind nearer LiDAR points in the camera
        Ps.append(P[keep]); Cs.append(img[v[keep], u[keep]])
    P, C = np.concatenate(Ps), np.concatenate(Cs).astype(np.float64)
    # one point per occupied voxel; position = centroid of all raw points in the 3x3x3 voxel block around it: the range
    # noise (sigma ~2.4 cm, along the ray) spreads a surface over a slab several cm thick, a plain per-voxel mean stays
    # biased off the surface (the voxel cuts the slab), the block centroid of a locally planar patch lies on the plane
    key = np.floor(P / a.vox).astype(np.int64)
    key -= key.min(0)
    M = key.max(0) + 3
    code = lambda k: ((k[:, 0] + 1) * M[1] + (k[:, 1] + 1)) * M[2] + (k[:, 2] + 1)
    ucode, inv, cnt = np.unique(code(key), return_inverse=True, return_counts=True)
    inv = inv.ravel()
    S = np.zeros((len(ucode), 3)); np.add.at(S, inv, P)
    rgb = np.zeros((len(ucode), 3)); np.add.at(rgb, inv, C)
    rgb = np.clip(np.round(rgb / cnt[:, None]), 0, 255).astype(np.uint8)
    uk = np.zeros((len(ucode), 3), np.int64); uk[inv] = key
    BS, BN = np.zeros_like(S), np.zeros(len(ucode))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = code(uk + np.array([dx, dy, dz]))
                j = np.searchsorted(ucode, c).clip(max=len(ucode) - 1)
                hit = ucode[j] == c
                BS[hit] += S[j[hit]]; BN[hit] += cnt[j[hit]]
    xyz = BS / BN[:, None]
    print(f'{len(Ps)} training frames: {n_all} first-echo points, {len(P)} coloured (in image, not hidden), '
          f'{len(xyz)} after {100 * a.vox:.0f} cm voxel averaging (median {np.median(cnt):.0f} samples / voxel, {np.median(BN):.0f} / block)')

    os.makedirs(a.out_root, exist_ok=True)
    for f in os.listdir(a.data_root):                        # symlink everything except the initialisation
        if f in ('sparse', 'envs'): continue
        dst = os.path.join(a.out_root, f)
        if not os.path.lexists(dst): os.symlink(os.path.realpath(os.path.join(a.data_root, f)), dst)
    sp = os.path.join(a.out_root, 'sparse', '0')
    os.makedirs(sp, exist_ok=True)
    src = os.path.realpath(os.path.join(a.data_root, 'sparse', '0'))
    for f in os.listdir(src):                                # cameras / images of the COLMAP model, points replaced
        if not f.startswith('points3D') and not os.path.lexists(os.path.join(sp, f)): os.symlink(os.path.join(src, f), os.path.join(sp, f))
    save_ply(os.path.join(sp, 'points3D.ply'), xyz.astype(np.float32), rgb)
    print(f'-> {sp}/points3D.ply')

    if a.gt_surface:
        from plyfile import PlyData
        from scipy.spatial import cKDTree
        g = PlyData.read(a.gt_surface)['vertex']; G = np.stack([g['x'], g['y'], g['z']], 1)
        d, _ = cKDTree(G).query(xyz)
        print(f'eval vs GT surface: median {100 * np.median(d):.2f} cm, p90 {100 * np.percentile(d, 90):.2f} cm, '
              f'> 5 cm {100 * (d > 0.05).mean():.2f}%', end='')
        if a.mirror:                                         # GT mirror plane: virtual points = off-surface, on it after mirroring
            c, nn = np.array(a.mirror[:3]), np.array(a.mirror[3:]) / np.linalg.norm(a.mirror[3:])
            q = xyz - 2 * ((xyz - c) @ nn)[:, None] * nn
            dq, _ = cKDTree(G).query(q)
            print(f', mirror-virtual {100 * ((d > 0.05) & (dq < 0.03)).mean():.2f}%', end='')
        print()


if __name__ == '__main__':
    main()
