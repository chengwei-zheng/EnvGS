# how wrong is the StableNormal prior on the mirror, scene B: angle between data/datasets/mirror_b/<scene>/normals and
# <scene>_gtnorm/normals (identical except GT mirror pixels, which hold the GT plane normal in camera space)
#     python mirror_prior_vs_gt_b.py [scene=b300]      (r300 = scene B_rot30)
import os, sys, cv2, numpy as np
D = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/datasets/mirror_b/'
rd = lambda p: (lambda n: n[..., ::-1].astype(np.float64) / (65535 if n.dtype == np.uint16 else 255) * 2 - 1)(cv2.imread(p, -1))
S = sys.argv[1] if len(sys.argv) > 1 else 'b300'
A, per = [], []
for c in sorted(os.listdir(f'{D}{S}/normals')):
    a, g = rd(f'{D}{S}/normals/{c}/000000.png'), rd(f'{D}{S}_gtnorm/normals/{c}/000000.png')
    m = np.abs(a - g).sum(-1) > 1e-3
    if m.sum() < 500: continue
    u = a[m] / np.linalg.norm(a[m], axis=-1, keepdims=True); v = g[m] / np.linalg.norm(g[m], axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1))); A.append(ang); per.append((int(c), np.median(ang), (ang < 15).mean(), m.sum()))
A = np.concatenate(A)
print(f'{len(per)} frames, {A.size} mirror px: median {np.median(A):.1f} deg  <5 {np.mean(A<5):.2f}  <15 {np.mean(A<15):.2f}  <30 {np.mean(A<30):.2f}  >60 {np.mean(A>60):.2f}')
h, e = np.histogram(A, bins=[0, 5, 10, 15, 30, 45, 60, 90, 180]); print('hist', dict(zip([f'{int(x)}-{int(y)}' for x, y in zip(e[:-1], e[1:])], np.round(h / A.size, 3))))
per.sort(key=lambda t: t[1]); print('best frames ', [(f, round(m, 1)) for f, m, _, _ in per[:5]]); print('worst frames', [(f, round(m, 1)) for f, m, _, _ in per[-5:]])
