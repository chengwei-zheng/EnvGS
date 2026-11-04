# c480 mirror PSNR by horizontal position on the convex mirror and by reflected distance (2026-11-04; user: the middle of the mirror
# is sharp, left / right are blurry). Per val frame: camera ray -> GT sphere cap (eval_npz grid 1296x972 = full-res pixels 1::2),
# mask == 3 pixels; u = signed horizontal coordinate on the mirror (along cross(axis, z), m), reflected distance from gt_layers
# depth_reflected (z-depth of the reflected hit -> ray length minus camera-to-mirror length). MSE pooled over frames per bin.
#     GT_PERSP=.../dataset_c480/persp python mirror_lr_bins.py <sphere json> <run> [<run> ...]
import glob, json, os, sys
import cv2, numpy as np
P = os.environ['GT_PERSP']; E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/corridor/envgs_'
S = json.load(open(sys.argv[1])); C, R, AX, AP = np.array(S['centre']), S['R'], np.array(S['axis']), S['aperture']
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U); V = np.cross(U, AX)
tj = json.load(open(P + '/transforms.json')); sub = lambda a: a[1::2, 1::2][:972, :1296]
ys, xs = np.mgrid[0:972, 0:1296] * 2 + 1
UB = [(-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6)]
DB = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]


def geo(fr):
    M = np.array(fr['transform_matrix']); o = M[:3, 3]
    dc = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1)
    dc /= np.linalg.norm(dc, axis=-1, keepdims=True); d = dc @ M[:3, :3].T
    oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R); t = -b - np.sqrt(np.maximum(disc, 0))
    X = o + t[..., None] * d - C
    zr = sub(np.load(P + '/' + fr['gt_layers_path'])['depth_reflected'])
    rd = zr / -dc[..., 2] - t
    return np.where(disc > 0, X @ U, np.nan), rd


cache = {}
res = {}
for r in sys.argv[2:]:
    su, nu, sd, nd = np.zeros(len(UB)), np.zeros(len(UB)), np.zeros(len(DB)), np.zeros(len(DB))
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
        if fid not in cache:
            g = cv2.resize(cv2.imread(P + '/' + fr['file_path'])[..., ::-1], (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            cache[fid] = (g, sub(cv2.imread(P + '/' + fr['mask_path'], -1)) == 3, *geo(fr))
        g, mm, u, rd = cache[fid]
        e = ((np.load(f)['rgb'].astype(np.float32) / 255 - g) ** 2).mean(-1)
        for k, (lo, hi) in enumerate(UB):
            m = mm & (u >= lo) & (u < hi); su[k] += e[m].sum(); nu[k] += m.sum()
        for k, (lo, hi) in enumerate(DB):
            m = mm & (rd >= lo) & (rd < hi); sd[k] += e[m].sum(); nd[k] += m.sum()
    ps = lambda s, n: ' | '.join(f'{-10 * np.log10(s[k] / n[k]):5.2f} ({int(n[k] / 1000)}k)' if n[k] else '   -   ' for k in range(len(s)))
    print(f'{r:40s} u (m) ' + ' '.join(f'{lo:+.1f}..{hi:+.1f}' for lo, hi in UB) + ':  ' + ps(su, nu))
    print(f'{"":40s} refl dist (m) ' + ' '.join(f'{lo}-{hi}' for lo, hi in DB) + ':  ' + ps(sd, nd), flush=True)
