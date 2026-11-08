# as mirror_lr_depth.py, right part split by whether the GT mirror point lies inside the LiDAR mirror extent (patch0.npz, optional
# dilation in cells): the msurf normal and the msurf_* consistency losses only act inside it (2026-11-07).
#     GT_PERSP=.../persp python mirror_lr_depth_extent.py <sphere json> <patch0.npz> <dilate cells> <run> [<run> ...]
import glob, json, os, sys
import cv2, numpy as np
from scipy import ndimage
P = os.environ['GT_PERSP']; E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/corridor/envgs_'
S = json.load(open(sys.argv[1])); C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
a = np.load(sys.argv[2]); ext = a['extent'].astype(bool); d = int(sys.argv[3])
if d: ext = ndimage.binary_dilation(ext, iterations=d)
tj = json.load(open(P + '/transforms.json')); sub = lambda x: x[1::2, 1::2][:972, :1296]
ys, xs = np.mgrid[0:972, 0:1296] * 2 + 1
UB = [(-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6)]
for r in sys.argv[4:]:
    acc = {k: [[] for _ in UB] for k in ('in', 'out')}
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
        M = np.array(fr['transform_matrix']); o = M[:3, 3]
        dc = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True); dd = dc @ M[:3, :3].T
        oc = o - C; b = dd @ oc; disc = b * b - (oc @ oc - R * R); t = -b - np.sqrt(np.maximum(disc, 0))
        X = o + t[..., None] * dd
        mm = (sub(cv2.imread(P + '/' + fr['mask_path'], -1)) == 3) & (disc > 0)
        rl = np.load(f)['depth'] / -dc[..., 2]
        L = (X - a['c']) @ a['R']; ij = np.floor((L[..., :2] - a['g0']) / a['cell']).astype(int)
        okij = (ij[..., 0] >= 0) & (ij[..., 0] < ext.shape[1]) & (ij[..., 1] >= 0) & (ij[..., 1] < ext.shape[0])
        inside = okij & ext[ij[..., 1].clip(0, ext.shape[0] - 1), ij[..., 0].clip(0, ext.shape[1] - 1)]
        u = (X - C) @ U
        for k, (lo, hi) in enumerate(UB):
            m = mm & (u >= lo) & (u < hi)
            acc['in'][k].append((rl - t)[m & inside]); acc['out'][k].append((rl - t)[m & ~inside])
    print(f'== {r} (extent dilated {d} cells)')
    print('u (m)                     | ' + ' | '.join(f'{lo:+.1f}..{hi:+.1f}' for lo, hi in UB))
    for k in ('in', 'out'):
        e = [np.concatenate(x) for x in acc[k]]
        print(f'{k:3s} extent: share of px    | ' + ' | '.join(f'{100 * len(x) / max(len(x) + len(np.concatenate(acc["out" if k == "in" else "in"][i])), 1):8.1f}%' for i, x in enumerate(e)))
        print(f'{k:3s} extent: med err (cm)   | ' + ' | '.join(f'{100 * np.median(x):9.2f}' if len(x) else '      -  ' for x in e))
        print(f'{k:3s} extent: |err| > 2 cm   | ' + ' | '.join(f'{100 * (np.abs(x) > .02).mean():8.1f}%' if len(x) else '      -  ' for x in e))
