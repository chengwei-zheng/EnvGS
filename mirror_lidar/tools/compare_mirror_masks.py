"""Compare per-camera mirror masks against the GT mirror (2026-11-09, c2_480: SAM-fused masks vs masks re-derived from the
voted surface extent; evaluation only, nothing written except the report / overlays).
Masks are given as name=<dir>[:mode]; <dir>/<cam>/000000.png. mode:
    in      pixels == 255 (the lidar_mirror_mask.py interior: what the normal loss / mirror mask uses)
    inband  pixels > 0 (interior + boundary band)
    any     pixels > 0 (plain binary masks, e.g. SAM output)
    fuse:<lidar mask dir>:<px>   LiDAR interior | (this mask within <px> of the LiDAR mask): the L+<variant> of sam_mirror_mask.py
Metrics: summed P / R / IoU, per-view mIoU, the right side of the mirror (u < --u-right via the GT sphere), and the mean
boundary error (px): for each view, the mean distance from the predicted contour to the GT contour, and missed / extra GT
pixels split by distance to the GT rim (< 10 px = rim, >= 10 px = gross).
    python mirror_lidar/tools/compare_mirror_masks.py <gt persp> <out dir> name=<dir>[:mode] ... [--sphere json] [--train-only]
"""
import argparse, glob, json, os
import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('gt'); ap.add_argument('out'); ap.add_argument('masks', nargs='+')
ap.add_argument('--sphere', default=''); ap.add_argument('--u-right', type=float, default=-0.4)
ap.add_argument('--train-only', action='store_true', help='only k % 8 != 0 (the EnvGS train views)')
ap.add_argument('--vis-step', type=int, default=24); ap.add_argument('--images', default='', help='<data_root>/images (overlays)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
gts = sorted(glob.glob(os.path.join(a.gt, 'masks', '*.png')))
tj = json.load(open(os.path.join(a.gt, 'transforms.json')))
S = json.load(open(a.sphere)) if a.sphere else None
specs = []
for m in a.masks:
    name, rest = m.split('=', 1)
    parts = rest.split(':')
    specs.append((name, parts[0], parts[1] if len(parts) > 1 else 'any', parts[2:]))
cams = sorted(os.listdir(specs[0][1]))


def u_map(k, H, W):
    C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
    U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
    M = np.array(tj['frames'][k]['transform_matrix']); o = M[:3, 3]
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones((H, W))], -1) @ M[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R)
    X = o + (-b - np.sqrt(np.maximum(disc, 0)))[..., None] * d - C
    return np.where(disc > 0, X @ U, np.nan)


def load(name, d, mode, extra, cam, shape):
    p = cv2.imread(os.path.join(d, cam, '000000.png'), cv2.IMREAD_UNCHANGED)
    if p is None:
        if mode != 'fuse': return np.zeros(shape, bool)
        p = np.zeros(shape, np.uint8)                                   # no SAM output for this view: LiDAR interior only
    if p.ndim == 3: p = p[..., 0]
    if mode == 'in': return p == 255
    if mode in ('inband', 'any'): return p > 0
    if mode == 'fuse':
        lm = cv2.imread(os.path.join(extra[0], cam, '000000.png'), cv2.IMREAD_UNCHANGED)
        if lm is None: return np.zeros(shape, bool)
        r = int(extra[1])
        dl = cv2.dilate((lm > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
        return (lm == 255) | ((p > 0) & dl)
    raise ValueError(mode)


acc = {n: {k: dict(tp=0, pred=0, gt=0, iou=[]) for k in ('all', 'right')} for n, *_ in specs}
bnd = {n: dict(cdist=[], miss_rim=0, miss_far=0, extra_rim=0, extra_far=0) for n, *_ in specs}
nview = 0
for k, cam in enumerate(cams):
    if a.train_only and k % 8 == 0: continue
    g = cv2.imread(gts[k], cv2.IMREAD_UNCHANGED); g = (g[..., 0] if g.ndim == 3 else g) == 3
    preds = {n: load(n, d, m, e, cam, g.shape) for n, d, m, e in specs}
    if not g.any() and not any(p.any() for p in preds.values()): continue
    nview += 1
    H, W = g.shape
    right = np.nan_to_num(u_map(k, H, W), nan=1e9) < a.u_right if S is not None else None
    # distance to the GT rim (both sides) and the GT contour distance field
    gu = g.astype(np.uint8)
    d_in = cv2.distanceTransform(gu, cv2.DIST_L2, 5); d_out = cv2.distanceTransform(1 - gu, cv2.DIST_L2, 5)
    drim = np.where(g, d_in, d_out)
    for n, p in preds.items():
        for key, reg in (('all', None), ('right', right)):
            if key == 'right' and reg is None: continue
            pp, gg = (p, g) if reg is None else (p & reg, g & reg)
            tp = int((pp & gg).sum()); s = acc[n][key]
            s['tp'] += tp; s['pred'] += int(pp.sum()); s['gt'] += int(gg.sum())
            if pp.any() or gg.any(): s['iou'].append(tp / max(int(pp.sum()) + int(gg.sum()) - tp, 1))
        b = bnd[n]
        miss, extra = g & ~p, p & ~g
        b['miss_rim'] += int((miss & (drim < 10)).sum()); b['miss_far'] += int((miss & (drim >= 10)).sum())
        b['extra_rim'] += int((extra & (drim < 10)).sum()); b['extra_far'] += int((extra & (drim >= 10)).sum())
        if p.any() and g.any():
            cs, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = np.concatenate(cs)[:, 0]
            inb = (c[:, 0] > 0) & (c[:, 0] < W - 1) & (c[:, 1] > 0) & (c[:, 1] < H - 1)   # image border is not a mask edge
            if inb.any(): b['cdist'].append(float(drim[c[inb, 1], c[inb, 0]].mean()))
    if a.vis_step and k % a.vis_step == 0 and g.any():
        img = cv2.imread(os.path.join(a.images, cam, '000000.png')) if a.images else None
        if img is None: img = np.full((H, W, 3), 80, np.uint8)
        ys, xs = np.nonzero(g | np.any([p for p in preds.values()], 0))
        cx, cy, s_ = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.3
        x0, y0 = int(max(cx - s_ / 2, 0)), int(max(cy - s_ / 2, 0)); x1, y1 = int(min(cx + s_ / 2, W)), int(min(cy + s_ / 2, H))
        sc = 480 / max(x1 - x0, y1 - y0); tw, th = int((x1 - x0) * sc), int((y1 - y0) * sc)
        rs = lambda m: cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
        base = cv2.resize(img[y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA); gc = rs(g)
        tiles = []
        for n, p in preds.items():
            t = base.astype(np.float32); pc = rs(p)
            t[pc & gc] = 0.6 * t[pc & gc] + 0.4 * np.array((0, 255, 0))
            t[pc & ~gc] = 0.4 * t[pc & ~gc] + 0.6 * np.array((255, 0, 0))     # extra: blue
            t[gc & ~pc] = 0.4 * t[gc & ~pc] + 0.6 * np.array((0, 0, 255))     # missed: red
            t = t.astype(np.uint8)
            cs, _ = cv2.findContours(gc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(t, cs, -1, (255, 0, 255), 1)
            tp = (p & g).sum()
            for i, s2 in enumerate([n, f'P {tp / max(p.sum(), 1) * 100:.1f} R {tp / g.sum() * 100:.1f}']):
                cv2.putText(t, s2, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(t, s2, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        cv2.imwrite(os.path.join(a.out, f'{cam}.jpg'), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])

lines = [f'{nview} views with a GT or predicted mirror' + (' (train views only)' if a.train_only else ''),
         f'{"mask":22s} | {"P":>6s} {"R":>6s} {"IoU":>6s} {"mIoU":>6s} | {"right P":>7s} {"R":>6s} {"IoU":>6s} | '
         f'{"edge px":>7s} | missed % of GT: rim / far | extra % of GT: rim / far']
res = {}
for n, *_ in specs:
    row = f'{n:22s} |'
    res[n] = {}
    for key in ('all', 'right'):
        s = acc[n][key]
        if key == 'right' and S is None: continue
        P, R = s['tp'] / max(s['pred'], 1), s['tp'] / max(s['gt'], 1); I = s['tp'] / max(s['pred'] + s['gt'] - s['tp'], 1)
        res[n][key] = dict(precision=P, recall=R, iou=I, miou=float(np.mean(s['iou'])) if s['iou'] else 0.0)
        row += (f' {P * 100:6.2f} {R * 100:6.2f} {I * 100:6.2f} {res[n][key]["miou"] * 100:6.2f} |' if key == 'all'
                else f' {P * 100:7.2f} {R * 100:6.2f} {I * 100:6.2f} |')
    b = bnd[n]; G = acc[n]['all']['gt']
    res[n]['boundary'] = dict(edge_px_mean=float(np.mean(b['cdist'])) if b['cdist'] else np.nan,
                              **{f: b[f] / max(G, 1) for f in ('miss_rim', 'miss_far', 'extra_rim', 'extra_far')})
    r = res[n]['boundary']
    row += (f' {r["edge_px_mean"]:7.2f} |       {r["miss_rim"] * 100:5.2f} / {r["miss_far"] * 100:5.2f}'
            f'       |        {r["extra_rim"] * 100:5.2f} / {r["extra_far"] * 100:5.2f}')
    lines.append(row)
print('\n'.join(lines))
open(os.path.join(a.out, 'report.txt'), 'w').write('\n'.join(lines) + '\n\n' + ' '.join(f'{n}={d}:{m}' for n, d, m, _ in specs) + '\n')
json.dump(res, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
