# LiDAR mirror masks (mirror_masks/<cam>/000000.png: 255 interior, 128 band) vs GT persp masks (== 3), binned by horizontal
# position u on the GT mirror (as mirror_lr_bins.py), every 4th camera, half resolution (2026-11-07).
#     GT_PERSP=.../persp python mask_lr_eval.py <sphere json> <mirror_masks dir>
import json, os, sys
import cv2, numpy as np
P = os.environ['GT_PERSP']
S = json.load(open(sys.argv[1])); C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
tj = json.load(open(P + '/transforms.json')); names = sorted(os.listdir(sys.argv[2]))
ys, xs = np.mgrid[0:972, 0:1296] * 2 + 1
UB = [(-0.7, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.7)]
n = np.zeros((len(UB), 4))   # GT px, GT & interior, GT & band, interior & ~GT (u from the sphere hit, if any)
fp_off = 0                   # interior pixels whose ray misses the GT sphere cap region entirely
for i in range(0, len(names), 4):
    fr = tj['frames'][i]; M = np.array(fr['transform_matrix']); o = M[:3, 3]
    g = cv2.imread(P + '/' + fr['mask_path'], -1)[1::2, 1::2][:972, :1296] == 3
    m = cv2.imread(os.path.join(sys.argv[2], names[i], '000000.png'), -1)[1::2, 1::2][:972, :1296]
    if not (g.any() or (m == 255).any()): continue
    d = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1) @ M[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R); t = -b - np.sqrt(np.maximum(disc, 0))
    u = np.where(disc > 0, (o + t[..., None] * d - C) @ U, np.nan)
    fp_off += ((m == 255) & ~g & np.isnan(u)).sum()
    for k, (lo, hi) in enumerate(UB):
        s = (u >= lo) & (u < hi)
        n[k] += [(g & s).sum(), (g & s & (m == 255)).sum(), (g & s & (m == 128)).sum(), (s & ~g & (m == 255)).sum()]
print('u (m)                  | ' + ' | '.join(f'{lo:+.1f}..{hi:+.1f}' for lo, hi in UB) + ' | all')
T = n.sum(0)
row = lambda name, f: print(f'{name:22s} | ' + ' | '.join(f'{f(x):8.1f}%' for x in n) + f' | {f(T):5.1f}%')
row('recall (interior)', lambda x: 100 * x[1] / max(x[0], 1))
row('in band (ignored)', lambda x: 100 * x[2] / max(x[0], 1))
row('missed', lambda x: 100 * (x[0] - x[1] - x[2]) / max(x[0], 1))
row('precision (interior)', lambda x: 100 * x[1] / max(x[1] + x[3], 1))
print(f'interior px not on the sphere at all: {fp_off} ({100 * fp_off / max(T[1] + T[3] + fp_off, 1):.2f}% of interior)')
