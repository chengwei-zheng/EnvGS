# LiDAR first-echo depth samples (LidarDepthSupervisor input) on the c2_480 mirror, by horizontal position u (as mirror_lr_bins.py):
# samples per bin (all 480 cameras) and their error vs the GT sphere (2026-11-06; right side of the mirror: Gaussians ~4 cm in front).
#     GT_PERSP=.../persp python mirror_lr_lidar.py <sphere json> <lidar_depth_dir>
import glob, json, os, sys
import cv2, numpy as np
P = os.environ['GT_PERSP']
S = json.load(open(sys.argv[1])); C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
tj = json.load(open(P + '/transforms.json'))
UB = [(-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6)]
names = sorted(f for f in os.listdir(sys.argv[2]) if f.endswith('.npz'))
E = [[] for _ in UB]
for i, f in enumerate(names):
    d = np.load(os.path.join(sys.argv[2], f)); h, w = int(d['H']), int(d['W'])
    fr = tj['frames'][i]; M = np.array(fr['transform_matrix']); o = M[:3, 3]
    s = tj['w'] / w
    y, x = np.divmod(d['idx'], w); xs, ys = (x + 0.5) * s, (y + 0.5) * s
    dc = np.stack([(xs - tj['cx']) / tj['fl_x'], -(ys - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1)
    dn = dc / np.linalg.norm(dc, axis=-1, keepdims=True); dd = dn @ M[:3, :3].T
    oc = o - C; b = dd @ oc; disc = b * b - (oc @ oc - R * R); t = -b - np.sqrt(np.maximum(disc, 0))
    mk = cv2.imread(P + '/' + fr['mask_path'], -1)
    m3 = mk[np.clip(ys.astype(int), 0, mk.shape[0] - 1), np.clip(xs.astype(int), 0, mk.shape[1] - 1)] == 3
    ok = m3 & (disc > 0)
    X = o + t[:, None] * dd; u = (X - C) @ U
    err = d['z'] - t * -dn[:, 2]                     # z-depth error, + = behind the mirror
    for k, (lo, hi) in enumerate(UB): E[k].append(err[ok & (u >= lo) & (u < hi)])
E = [np.concatenate(e) for e in E]
print('u (m)              | ' + ' | '.join(f'{lo:+.1f}..{hi:+.1f}' for lo, hi in UB))
print('samples (all cams) | ' + ' | '.join(f'{len(e):9d}' for e in E))
print('med err (cm,+far)  | ' + ' | '.join(f'{100 * np.median(e):9.2f}' for e in E))
print('|err| < 5 cm       | ' + ' | '.join(f'{100 * (np.abs(e) < .05).mean():8.1f}%' for e in E))
print('err > +5 cm (ghost)| ' + ' | '.join(f'{100 * (e > .05).mean():8.1f}%' for e in E))
