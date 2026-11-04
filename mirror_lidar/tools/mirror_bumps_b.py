"""Bumps on the scene-B mirror: depth-normal consistency of EnvGS mirror_b runs on the val views (2026-10-27).

Uses the NORMAL (rendered surfel normal) and SURFACE_NORMAL (normal from the rendered depth) images written by the final
evaluation (data/result/.../NORMAL, SURFACE_NORMAL, frame0000_camera????.png, 8-bit, (n + 1) / 2, ~0.2-0.4 deg quantisation).
Per pixel gs = 1 - n_render . n_depth (the gs_norm term, without the depth scale) and the angle between them, on the GT
mirror pixels (persp mask 3, eroded by 9 px against the rim) and on the rest (dilated mirror excluded). Bumps show up as
the tail: fraction of mirror pixels above 2 / 5 / 10 deg.

    python mirror_bumps_b.py <run> [<run> ...]     (run = exp tag without envgs_; writes <result>/mirror_bumps.json)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/' + os.environ.get('RESULT_GROUP', 'mirror_b') + '/envgs_'  # RESULT_GROUP=corridor for c480 (2026-10-29)
P = os.environ.get('GT_PERSP', '/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300_lamppanel/persp')
tj = json.load(open(P + '/transforms.json'))


def dec(p):
    v = cv2.imread(p)[..., ::-1].astype(np.float32) / 255 * 2 - 1
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


print(f'{"run":46s} | mirror: {"1-cos":>8s} {"med":>6s} {">2°":>6s} {">5°":>6s} {">10°":>6s} | rest: {"1-cos":>7s} | views')
for r in sys.argv[1:]:
    Mg, Og = [], []
    fs = sorted(glob.glob(f'{E}{r}/NORMAL/frame0000_camera????.png'))
    for f in fs:
        fr = tj['frames'][int(f[-8:-4])]
        n, s = dec(f), dec(f.replace('/NORMAL/', '/SURFACE_NORMAL/'))
        m = (cv2.imread(P + '/' + fr['mask_path'], -1)[1::2, 1::2][:n.shape[0], :n.shape[1]] == 3).astype(np.uint8)
        k = np.ones((19, 19), np.uint8)
        g = 1 - (n * s).sum(-1)
        Mg.append(g[cv2.erode(m, k) > 0]); Og.append(g[cv2.dilate(m, k) == 0])
    mg, og = np.concatenate(Mg), np.concatenate(Og)
    ang = np.degrees(np.arccos(np.clip(1 - mg, -1, 1)))
    row = dict(run=r, views=len(fs), mirror_gs=float(mg.mean()), mirror_ang_med=float(np.median(ang)),
               mirror_gt2=float(100 * (ang > 2).mean()), mirror_gt5=float(100 * (ang > 5).mean()), mirror_gt10=float(100 * (ang > 10).mean()),
               rest_gs=float(og.mean()))
    json.dump(row, open(f'{E}{r}/mirror_bumps.json', 'w'), indent=1)
    print(f'{r:46s} |         {row["mirror_gs"]:.5f} {row["mirror_ang_med"]:5.2f}° {row["mirror_gt2"]:5.1f}% {row["mirror_gt5"]:5.2f}% {row["mirror_gt10"]:5.2f}% '
          f'|       {row["rest_gs"]:.4f} | {len(fs)}', flush=True)
