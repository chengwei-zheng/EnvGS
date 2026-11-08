# c2_480 mirror pixels by horizontal position u on the mirror (as mirror_lr_bins.py): rendered (Gaussian) depth vs the GT sphere hit,
# fraction beyond the msurf gate (|t - dpt| < tol = 0.15 m, otherwise the pixel falls back to the 2DGS normal), and the incidence angle
# (2026-11-06; user: right side of the mirror (u < 0) still blurry while the normals of the spline are fine there).
#     GT_PERSP=.../persp python mirror_lr_depth.py <sphere json> <run> [<run> ...]
import glob, json, os, sys
import cv2, numpy as np
P = os.environ['GT_PERSP']; E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/corridor/envgs_'
S = json.load(open(sys.argv[1])); C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
tj = json.load(open(P + '/transforms.json')); sub = lambda a: a[1::2, 1::2][:972, :1296]
ys, xs = np.mgrid[0:972, 0:1296] * 2 + 1
UB = [(-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6)]
for r in sys.argv[2:]:
    acc = {k: [[] for _ in UB] for k in ('err', 'inc')}
    zdep = None
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
        M = np.array(fr['transform_matrix']); o = M[:3, 3]
        dc = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True); d = dc @ M[:3, :3].T
        oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R); t = -b - np.sqrt(np.maximum(disc, 0))
        X = o + t[..., None] * d; n = (X - C) / R
        mm = (sub(cv2.imread(P + '/' + fr['mask_path'], -1)) == 3) & (disc > 0)
        dep = np.load(f)['depth']
        if zdep is None:   # rendered depth: z-depth or ray length? pick the one matching the sphere on the mirror
            zdep = np.median(np.abs(dep[mm] - t[mm] * -dc[..., 2][mm])) < np.median(np.abs(dep[mm] - t[mm]))
        rl = dep / -dc[..., 2] if zdep else dep
        u = (X - C) @ U; inc = np.degrees(np.arccos(np.clip(-(d * n).sum(-1), 0, 1)))
        for k, (lo, hi) in enumerate(UB):
            m = mm & (u >= lo) & (u < hi)
            acc['err'][k].append((rl - t)[m]); acc['inc'][k].append(inc[m])
    print(f'== {r} (rendered depth is {"z-depth" if zdep else "ray length"})')
    print('u (m)            | ' + ' | '.join(f'{lo:+.1f}..{hi:+.1f}' for lo, hi in UB))
    E_ = [np.concatenate(x) for x in acc['err']]; I_ = [np.concatenate(x) for x in acc['inc']]
    print('med |err| (cm)   | ' + ' | '.join(f'{100 * np.median(np.abs(e)):9.2f}' for e in E_))
    print('med err (cm,+far)| ' + ' | '.join(f'{100 * np.median(e):9.2f}' for e in E_))
    print('|err| > 2 cm     | ' + ' | '.join(f'{100 * (np.abs(e) > .02).mean():8.1f}%' for e in E_))
    print('|err| > 15 cm    | ' + ' | '.join(f'{100 * (np.abs(e) > .15).mean():8.1f}%' for e in E_) + '   (msurf gate fails -> 2DGS normal)')
    print('incidence med (°)| ' + ' | '.join(f'{np.median(i):9.1f}' for i in I_), flush=True)
