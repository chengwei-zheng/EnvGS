"""Curved (spherical) mirror normal accuracy of EnvGS runs (2026-10-29, corridor c480; the curved-mirror counterpart of
mirror_normal_spread_b.py / mirror_flatness_b.py, which assume one plane).

On the GT mirror pixels of the val views (persp mask 3, eroded by 9 px at the 1296x972 eval resolution), the GT normal is
the sphere normal (X - c) / R at the GT depth point X. Compared with
  NORMAL          rendered surfel normal (8-bit (n + 1) / 2, OpenGL camera frame: n_cam = c2w^T n_world, checked on f300),
  SURFACE_NORMAL  normal of the rendered depth (same frame),
both written by the final evaluation into data/result/envgs/<group>/envgs_<run>/. Reports median / p90 and the fraction
above 2 / 5 / 10 deg, over all mirror pixels of all views, plus the per-view median.

    RESULT_GROUP=corridor GT_DS=<dataset_c480> MIRROR_SPHERE=-0.02669,2.42669,2.25706,1.0 python mirror_normal_sphere.py <run> ...
(gsplat env, for OpenEXR)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom_eval_b import read_depth, backproject  # noqa: E402

E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/' + os.environ.get('RESULT_GROUP', 'corridor') + '/envgs_'
DS = os.environ['GT_DS']
C = np.array([float(v) for v in os.environ['MIRROR_SPHERE'].split(',')])
SC, SR = C[:3], C[3]
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
tj = json.load(open(DS + '/persp/transforms.json'))
sub = lambda a: a[1::2, 1::2][:972, :1296]


def dec(p):
    v = cv2.imread(p)[..., ::-1].astype(np.float32) / 255 * 2 - 1
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def stats(a):
    return dict(med=float(np.median(a)), p90=float(np.percentile(a, 90)), gt2=float(100 * (a > 2).mean()),
                gt5=float(100 * (a > 5).mean()), gt10=float(100 * (a > 10).mean()))


print(f'{"run":52s} | {"":8s} {"med":>6s} {"p90":>6s} {">2°":>6s} {">5°":>6s} {">10°":>6s} {"view med":>8s} | views')
for r in sys.argv[1:]:
    acc = {'NORMAL': [], 'SURFACE_NORMAL': []}
    vmed = {'NORMAL': [], 'SURFACE_NORMAL': []}
    fs = sorted(glob.glob(f'{E}{r}/NORMAL/frame0000_camera????.png'))
    nv = 0
    for f in fs:
        fr = tj['frames'][int(f[-8:-4])]
        m = (sub(cv2.imread(DS + '/persp/' + fr['mask_path'], -1)) == 3).astype(np.uint8)
        m = cv2.erode(m, np.ones((19, 19), np.uint8)) > 0
        if m.sum() < 500: continue
        nv += 1
        c2w_gl = np.array(fr['transform_matrix'])
        z = sub(read_depth(DS + '/persp/' + fr['depth_file_path'], tj['h'], tj['w'])).astype(np.float64)
        X = backproject(np.where(m, z, 0), c2w_gl @ GL_TO_CV, tj)[m]
        gw = (X - SC) / np.linalg.norm(X - SC, axis=1, keepdims=True)
        gc = gw @ c2w_gl[:3, :3]                                   # world -> OpenGL camera (c2w^T n)
        for k in acc:
            n = dec(f.replace('/NORMAL/', f'/{k}/'))[m]
            a = np.degrees(np.arccos(np.clip(np.abs((n * gc).sum(1)), 0, 1)))
            acc[k].append(a); vmed[k].append(np.median(a))
    row = dict(run=r, views=nv)
    for k in acc:
        row[k] = dict(**stats(np.concatenate(acc[k])), view_med=float(np.median(vmed[k])))
        s = row[k]
        print(f'{r:52s} | {k[:8]:8s} {s["med"]:5.2f}° {s["p90"]:5.2f}° {s["gt2"]:5.1f}% {s["gt5"]:5.1f}% {s["gt10"]:5.2f}% {s["view_med"]:7.2f}° | {nv}', flush=True)
    json.dump(row, open(f'{E}{r}/mirror_normal_sphere.json', 'w'), indent=1)
