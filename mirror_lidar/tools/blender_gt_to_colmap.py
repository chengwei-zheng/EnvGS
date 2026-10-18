"""Blender synthetic dataset (persp/transforms.json, GT poses) -> COLMAP sparse model with GT poses.

Poses are NOT estimated: we only extract/match features and triangulate (colmap point_triangulator)
with the GT cameras fixed, so the sparse points are RGB-only (no LiDAR leaks into the baseline's init)
and the world frame is the Blender world (metric, Z up).

    python blender_gt_to_colmap.py <dataset_dir> <out_dir> [--colmap BIN] [--gpu 0]
      <out_dir>/images -> <dataset_dir>/persp/rgb (symlink)
      <out_dir>/database.db, <out_dir>/gt_model/ (txt, no points), <out_dir>/sparse/0/ (triangulated, bin)

2026-10-14, first used for proj_indoor_mirror_a/dataset_a220.
"""
import argparse
import json
import os
import sqlite3
import subprocess

import numpy as np


def rotmat2qvec(R):
    # same as colmap's read_write_model.rotmat2qvec
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                  [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def run(cmd):
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('out')
    ap.add_argument('--colmap', default=os.path.expanduser('~/miniconda3/envs/colmap_cuda/bin/colmap'))
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()

    T = json.load(open(os.path.join(a.dataset, 'persp', 'transforms.json')))
    assert T['camera_model'] == 'OPENCV' and not any(T[k] for k in ('k1', 'k2', 'p1', 'p2')), 'expects undistorted pinhole'
    W, H, fx, fy, cx, cy = T['w'], T['h'], T['fl_x'], T['fl_y'], T['cx'], T['cy']
    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(a.out, 'images')
    if not os.path.exists(img_dir):
        os.symlink(os.path.abspath(os.path.join(a.dataset, 'persp', 'rgb')), img_dir)
    # name -> world-to-camera (OpenCV: +X right, +Y down, +Z forward); Blender c2w is -Z forward, +Y up
    poses = {}
    for f in T['frames']:
        name = os.path.basename(f['file_path'])
        if not name.endswith('.png'):
            name += '.png'
        c2w = np.array(f['transform_matrix'], dtype=np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])
        poses[name] = np.linalg.inv(c2w)
    assert all(os.path.exists(os.path.join(img_dir, n)) for n in poses), 'missing images'

    db = os.path.join(a.out, 'database.db')
    if os.path.exists(db):
        os.remove(db)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    run([a.colmap, 'feature_extractor', '--database_path', db, '--image_path', img_dir,
         '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera', '1',
         '--ImageReader.camera_params', f'{fx},{fy},{cx},{cy}'])
    run([a.colmap, 'exhaustive_matcher', '--database_path', db])

    con = sqlite3.connect(db)
    ids = {n: i for i, n in con.execute('SELECT image_id, name FROM images')}
    cam_id = con.execute('SELECT camera_id FROM cameras').fetchall()
    con.close()
    assert len(cam_id) == 1 and set(ids) == set(poses), (len(cam_id), len(ids), len(poses))
    cam_id = cam_id[0][0]

    gt = os.path.join(a.out, 'gt_model')
    os.makedirs(gt, exist_ok=True)
    with open(os.path.join(gt, 'cameras.txt'), 'w') as f:
        f.write(f'{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n')
    with open(os.path.join(gt, 'images.txt'), 'w') as f:
        for n in sorted(poses):
            w2c = poses[n]
            q, t = rotmat2qvec(w2c[:3, :3]), w2c[:3, 3]
            f.write(f'{ids[n]} {" ".join(map(str, q))} {" ".join(map(str, t))} {cam_id} {n}\n\n')
    open(os.path.join(gt, 'points3D.txt'), 'w').close()

    sp = os.path.join(a.out, 'sparse', '0')
    os.makedirs(sp, exist_ok=True)
    run([a.colmap, 'point_triangulator', '--database_path', db, '--image_path', img_dir,
         '--input_path', gt, '--output_path', sp,
         '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_principal_point', '0',
         '--Mapper.ba_refine_extra_params', '0'])
    run([a.colmap, 'model_analyzer', '--path', sp])


if __name__ == '__main__':
    main()
