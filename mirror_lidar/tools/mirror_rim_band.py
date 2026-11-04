# mirror PSNR by distance from the rim (2026-11-03, scene B_convex / cv300): per val frame, the GT sphere cap is intersected per pixel
# (eval_npz 1296x972 = full-res pixels 1::2 as mirror_perc_b.py) and the mirror pixels (mask == 3) are binned by aperture - radial
# distance from the mirror axis; MSE is pooled over all frames per bin. frame_hit_stats.py: 96% / 50% / 18% / 1% of the reflected rays
# in the 1-2 / 2-3 / 3-5 / 5-10 cm bins hit the frame within START_OFFSET = 6 cm.
#     GT_PERSP=.../dataset_cv300/persp python mirror_rim_band.py <sphere json> <run> [<run> ...]
import glob, json, os, sys
import cv2, numpy as np
P = os.environ['GT_PERSP']; E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/mirror_b/envgs_'
S = json.load(open(sys.argv[1])); C, R, AX, AP = np.array(S['centre']), S['R'], np.array(S['axis']), S['aperture']
tj = json.load(open(P + '/transforms.json')); sub = lambda a: a[1::2, 1::2][:972, :1296]
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 60)]
ys, xs = np.mgrid[0:972, 0:1296] * 2 + 1


def rim_dist(fr):
    M = np.array(fr['transform_matrix']); o = M[:3, 3]
    d = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones(xs.shape)], -1) @ M[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R)
    X = o + (-b - np.sqrt(np.maximum(disc, 0)))[..., None] * d - C
    rad = np.linalg.norm(X - (X @ AX)[..., None] * AX, axis=-1)
    return np.where(disc > 0, AP - rad, np.nan)


print(f'{"run":44s} | ' + ' | '.join(f'{lo}-{hi} cm' for lo, hi in BINS))
cache = {}
for r in sys.argv[2:]:
    se = np.zeros(len(BINS)); n = np.zeros(len(BINS))
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
        if fid not in cache:
            g = cv2.resize(cv2.imread(P + '/' + fr['file_path'])[..., ::-1], (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            cache[fid] = (g, sub(cv2.imread(P + '/' + fr['mask_path'], -1)) == 3, rim_dist(fr))
        g, mm, rd = cache[fid]
        e = ((np.load(f)['rgb'].astype(np.float32) / 255 - g) ** 2).mean(-1)
        for k, (lo, hi) in enumerate(BINS):
            m = mm & (rd >= lo / 100) & (rd < hi / 100); se[k] += e[m].sum(); n[k] += m.sum()
    print(f'{r:44s} | ' + ' | '.join(f'{-10 * np.log10(se[k] / n[k]):5.2f} ({int(n[k] / 1000)}k)' if n[k] else '   -   ' for k in range(len(BINS))), flush=True)
