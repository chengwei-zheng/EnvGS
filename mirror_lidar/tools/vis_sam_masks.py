"""Zoomed comparison of mirror masks (2026-11-09, SAM mask refinement): for every --step-th camera with a mirror, a crop around the
LiDAR / GT mirror (padded by --pad of its size, resized to --tile px) per mask: green fill = mask, magenta = GT outline (evaluation
only), yellow = LiDAR outline (interior + band), red = GT mirror missed by the mask, label = precision / recall vs GT.
Masks: <run dir>/<variant>/<cam>/000000.png (sam_mirror_mask.py), plus 'lidar' (the LiDAR interior) and 'L+<variant>'
(LiDAR interior | SAM within --fuse-px of the LiDAR mask).
    python mirror_lidar/tools/vis_sam_masks.py <data_root> <lidar mirror_masks> <gt persp> <out dir> <variant dir>:<variant>[,...] ...
"""
import argparse, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('data_root'); ap.add_argument('masks'); ap.add_argument('gt'); ap.add_argument('out')
ap.add_argument('runs', nargs='+', help='<run dir>:<variant>[,<variant>...]; L+<variant> for the LiDAR fusion')
ap.add_argument('--step', type=int, default=6); ap.add_argument('--pad', type=float, default=0.25)
ap.add_argument('--tile', type=int, default=480); ap.add_argument('--fuse-px', type=int, default=40); ap.add_argument('--cols', type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cams = sorted(os.listdir(os.path.join(a.data_root, 'images')))
gts = sorted(glob.glob(os.path.join(a.gt, 'masks', '*.png')))
specs = [(r.rsplit(':', 1)[0], v) for r in a.runs for v in r.rsplit(':', 1)[1].split(',')]
n = 0
for k in range(0, len(cams), a.step):
    cam = cams[k]
    g = cv2.imread(gts[k], cv2.IMREAD_UNCHANGED); g = (g[..., 0] if g.ndim == 3 else g) == 3
    lm = cv2.imread(os.path.join(a.masks, cam, '000000.png'), cv2.IMREAD_UNCHANGED)
    if lm is None or not (lm > 0).any() or not g.any(): continue
    img = cv2.imread(os.path.join(a.data_root, 'images', cam, '000000.png'))
    H, W = g.shape
    ys, xs = np.nonzero(g | (lm > 0))
    cx, cy, s = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * a.pad)
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)); x1, y1 = int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
    dl = cv2.dilate((lm > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.fuse_px + 1,) * 2)) > 0
    masks = [('lidar', lm == 255)]
    for run, v in specs:
        base = v[2:] if v.startswith('L+') else v
        p = cv2.imread(os.path.join(run, base, cam, '000000.png'), 0)
        if p is None: continue
        p = p > 0
        masks.append((v, (lm == 255) | (p & dl) if v.startswith('L+') else p))
    crop = lambda x: x[y0:y1, x0:x1]
    sc = a.tile / max(x1 - x0, y1 - y0); tw, th = int(round((x1 - x0) * sc)), int(round((y1 - y0) * sc))
    rs = lambda x: cv2.resize(crop(x).astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    base_img = cv2.resize(crop(img), (tw, th), interpolation=cv2.INTER_AREA)
    gc, lc = rs(g), rs(lm > 0)
    tiles = []
    for name, p in masks:
        t = base_img.astype(np.float32); pc = rs(p)
        t[pc] = 0.65 * t[pc] + 0.35 * np.array((0, 255, 0)); miss = gc & ~pc
        t[miss] = 0.4 * t[miss] + 0.6 * np.array((0, 0, 255)); t = t.astype(np.uint8)
        for m, col, w in ((gc, (255, 0, 255), 2), (lc, (0, 255, 255), 1)):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(t, cs, -1, col, w)
        tp = (p & g).sum()
        for i, s_ in enumerate([name, f'P {tp / max(p.sum(), 1) * 100:.1f}  R {tp / g.sum() * 100:.1f}']):
            cv2.putText(t, s_, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(t, s_, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % a.cols: tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + a.cols], 1) for i in range(0, len(tiles), a.cols)], 0)
    head = np.zeros((32, grid.shape[1], 3), np.uint8)
    cv2.putText(head, f'cam {cam}  mirror {g.mean() * 100:.1f}% of image  (crop {x1 - x0}x{y1 - y0} px)', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, f'{cam}.jpg'), np.concatenate([head, grid], 0), [cv2.IMWRITE_JPEG_QUALITY, 88]); n += 1
print(n, 'images ->', a.out)
