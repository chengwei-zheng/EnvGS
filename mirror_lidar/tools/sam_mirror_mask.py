"""Mirror masks from SAM 3 prompted with the LiDAR masks (2026-11-09, c2_480, step 1 of the SAM mask refinement: per image, no
propagation, no 3D voting). Run in the sam3 conda env (~/SMBC/SAM3/sam3, weights facebook/sam3 from the HF cache).

For every camera with a LiDAR mirror mask (lidar_mirror_mask.py format: 255 interior, 128 boundary band), prompt variants:
  box       box around the LiDAR mask (interior + band), padded by --box-pad of its size on every side
  box_pos   box + --npos positive points spread over the LiDAR interior eroded by --pos-erode px (farthest point sampling)
  box_pn    box_pos + --nneg negative points on a ring --neg-gap..--neg-gap+20 px outside the LiDAR mask (inside the padded box)
  text      text prompt --text (SAM 3 concept segmentation, no LiDAR prompt); the instance with the largest IoU with the LiDAR mask
SAM point/box prompts (predict_inst) return 3 candidates; the one with the highest predicted score is kept.
Output <out>/<variant>/<cam>/000000.png (255 = mirror) + <out>/metrics.json / metrics.txt; with --gt <blender persp dir> also
precision / recall / IoU against the GT mirror (persp/masks == 3, evaluation only), overall and on the right side of the mirror
(u < --u-right m along the horizontal mirror axis; needs --sphere, the GT mirror_sphere.json, evaluation only), and overlays for
every --vis-step-th camera in <out>/vis (green = SAM, magenta = GT outline, yellow = LiDAR outline, dots = point prompts).
    python mirror_lidar/tools/sam_mirror_mask.py <data_root> <lidar mirror_masks> <out> [--gt <persp>] [--sphere <json>]
"""
import argparse, glob, json, os, sys, time
import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('data_root'); ap.add_argument('masks'); ap.add_argument('out')
ap.add_argument('--gt', default=''); ap.add_argument('--sphere', default=''); ap.add_argument('--u-right', type=float, default=-0.4)
ap.add_argument('--variants', default='box,box_pos,box_pn,text')
ap.add_argument('--box-pad', type=float, default=0.05)
ap.add_argument('--npos', type=int, default=5); ap.add_argument('--pos-erode', type=int, default=25)
ap.add_argument('--nneg', type=int, default=6); ap.add_argument('--neg-gap', type=int, default=40)
ap.add_argument('--text', default='mirror')
ap.add_argument('--vis-step', type=int, default=24)
ap.add_argument('--sam3-root', default=os.path.expanduser('~/SMBC/SAM3/sam3'))
a = ap.parse_args()
VARIANTS = a.variants.split(',')

sys.path.insert(0, a.sam3_root)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
model = build_sam3_image_model(bpe_path=f'{a.sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz', enable_inst_interactivity=True)
proc = Sam3Processor(model)

cams = sorted(os.listdir(os.path.join(a.data_root, 'images')))
gt_files = sorted(glob.glob(os.path.join(a.gt, 'masks', '*.png'))) if a.gt else None
if gt_files: assert len(gt_files) == len(cams), (len(gt_files), len(cams))
tj = json.load(open(os.path.join(a.gt, 'transforms.json'))) if a.gt else None
S = json.load(open(a.sphere)) if a.sphere else None


def u_map(k, H, W):
    """horizontal coordinate u (m) of the GT sphere hit along the mirror's horizontal axis, nan off the sphere (evaluation only)"""
    C, R, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
    U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
    M = np.array(tj['frames'][k]['transform_matrix']); o = M[:3, 3]
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones((H, W))], -1) @ M[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = o - C; b = d @ oc; disc = b * b - (oc @ oc - R * R)
    X = o + (-b - np.sqrt(np.maximum(disc, 0)))[..., None] * d - C
    return np.where(disc > 0, X @ U, np.nan)


def fps(pts, n, rng):
    """farthest point sampling of n points (x, y) from pts"""
    if len(pts) <= n: return pts
    sel = [rng.integers(len(pts))]
    d = np.full(len(pts), np.inf)
    for _ in range(n - 1):
        d = np.minimum(d, ((pts - pts[sel[-1]]) ** 2).sum(-1)); sel.append(int(d.argmax()))
    return pts[sel]


def prompts(lm, rng):
    """box (x0, y0, x1, y1), positive and negative points (x, y) from the LiDAR mask"""
    H, W = lm.shape
    ys, xs = np.nonzero(lm > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = a.box_pad * (x1 - x0), a.box_pad * (y1 - y0)
    box = np.array([max(x0 - px, 0), max(y0 - py, 0), min(x1 + px, W - 1), min(y1 + py, H - 1)], np.float32)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode((lm == 255).astype(np.uint8), k(a.pos_erode))
    if not inner.any(): inner = (lm == 255).astype(np.uint8)
    pos = fps(np.stack(np.nonzero(inner)[::-1], -1), a.npos, rng) if inner.any() else np.zeros((0, 2), int)
    m = (lm > 0).astype(np.uint8)
    ring = cv2.dilate(m, k(a.neg_gap + 20)) & ~cv2.dilate(m, k(a.neg_gap)).astype(bool)
    bx = np.zeros_like(m, bool); bx[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1] = True
    ring = ring.astype(bool) & bx
    neg = fps(np.stack(np.nonzero(ring)[::-1], -1), a.nneg, rng) if ring.any() else np.zeros((0, 2), int)
    return box, pos, neg


def best(masks, scores):
    masks = masks.cpu().numpy() if torch.is_tensor(masks) else masks
    scores = scores.float().cpu().numpy() if torch.is_tensor(scores) else scores
    return masks[int(np.argmax(scores))] > 0


def stats(pred, gt, region=None):
    if region is not None: pred, gt = pred & region, gt & region
    tp = (pred & gt).sum()
    return dict(tp=int(tp), pred=int(pred.sum()), gt=int(gt.sum()))


acc = {}
def add(name, key, s):
    d = acc.setdefault(name, {}).setdefault(key, dict(tp=0, pred=0, gt=0, iou=[]))
    for f in ('tp', 'pred', 'gt'): d[f] += s[f]
    if s['gt'] or s['pred']: d['iou'].append(s['tp'] / max(s['pred'] + s['gt'] - s['tp'], 1))


for v in VARIANTS: os.makedirs(os.path.join(a.out, v), exist_ok=True)
os.makedirs(os.path.join(a.out, 'vis'), exist_ok=True)
rng = np.random.default_rng(0)
t0 = time.time(); n_done = n_noprompt = 0
for k, cam in enumerate(cams):
    lm = cv2.imread(os.path.join(a.masks, cam, '000000.png'), cv2.IMREAD_UNCHANGED)
    g = None
    if gt_files:
        g = cv2.imread(gt_files[k], cv2.IMREAD_UNCHANGED); g = (g[..., 0] if g.ndim == 3 else g) == 3
    has_l = lm is not None and (lm > 0).any()
    if not has_l and (g is None or not g.any()): continue
    img = cv2.imread(os.path.join(a.data_root, 'images', cam, '000000.png'))
    H, W = img.shape[:2]
    if lm is None: lm = np.zeros((H, W), np.uint8)
    state = proc.set_image(Image.fromarray(img[..., ::-1]))
    preds = {'lidar': lm == 255, 'lidar_band': lm > 0}
    box = pos = neg = None
    if has_l:
        box, pos, neg = prompts(lm, rng)
        for v in VARIANTS:
            if v == 'text': continue
            pc = pl = None
            if v in ('box_pos', 'box_pn') and len(pos):
                pc, pl = pos, np.ones(len(pos), int)
                if v == 'box_pn' and len(neg): pc, pl = np.concatenate([pos, neg]), np.concatenate([pl, np.zeros(len(neg), int)])
            m, s, _ = model.predict_inst(state, point_coords=None if pc is None else pc.astype(np.float32), point_labels=pl,
                                         box=box[None], multimask_output=True)
            preds[v] = best(m, s)
    else:
        n_noprompt += 1
    if 'text' in VARIANTS:
        out = proc.set_text_prompt(state=state, prompt=a.text)
        ms = out['masks']; ms = (ms.cpu().numpy() if torch.is_tensor(ms) else ms).reshape(-1, H, W) > 0
        ref = lm > 0
        if len(ms) == 0: preds['text'] = np.zeros((H, W), bool)
        elif ref.any(): preds['text'] = ms[int(np.argmax([(m & ref).sum() / max((m | ref).sum(), 1) for m in ms]))]
        else: preds['text'] = ms[int(np.argmax(out['scores'].float().cpu().numpy()))]
    for v in VARIANTS:
        if v not in preds: preds[v] = np.zeros((H, W), bool)   # no LiDAR prompt in this view
        d = os.path.join(a.out, v, cam); os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, '000000.png'), preds[v].astype(np.uint8) * 255)
    if g is not None:
        right = None
        if S is not None:
            u = u_map(k, H, W); right = np.nan_to_num(u, nan=1e9) < a.u_right
        for name, p in preds.items():
            add(name, 'all', stats(p, g))
            if right is not None: add(name, 'right', stats(p, g, right))
    if a.vis_step and k % a.vis_step == 0:
        sc = 0.5; vis = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        tiles = []
        for v in ['lidar'] + VARIANTS:
            t = vis.copy().astype(np.float32); p = cv2.resize(preds[v].astype(np.uint8), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            t[p] = 0.6 * t[p] + 0.4 * np.array((0, 255, 0)); t = t.astype(np.uint8)
            if g is not None:
                gg = cv2.resize(g.astype(np.uint8), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(gg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(t, cs, -1, (255, 0, 255), 2)
            ll = cv2.resize((lm > 0).astype(np.uint8), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(ll, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(t, cs, -1, (0, 255, 255), 1)
            if v != 'lidar' and v != 'text' and box is not None:
                cv2.rectangle(t, tuple((box[:2] * sc).astype(int)), tuple((box[2:] * sc).astype(int)), (255, 255, 0), 1)
                if v in ('box_pos', 'box_pn'):
                    for x, y in pos: cv2.circle(t, (int(x * sc), int(y * sc)), 5, (0, 200, 0), -1)
                if v == 'box_pn':
                    for x, y in neg: cv2.circle(t, (int(x * sc), int(y * sc)), 5, (0, 0, 255), -1)
            s = f'{v}'
            if g is not None and g.any():
                tp = (preds[v] & g).sum(); s += f'  P {tp / max(preds[v].sum(), 1) * 100:.1f}  R {tp / g.sum() * 100:.1f}'
            cv2.putText(t, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(t, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
        cv2.imwrite(os.path.join(a.out, 'vis', f'{cam}.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    n_done += 1
    if n_done % 20 == 0: print(f'{n_done} views ({time.time() - t0:.0f} s)', flush=True)

res = {}
lines = [f'{n_done} views ({n_noprompt} with GT mirror but no LiDAR mask: prompt variants empty there)',
         f'{"mask":12s} | {"all: P":>8s} {"R":>6s} {"IoU":>6s} {"mIoU":>6s} | {"right: P":>8s} {"R":>6s} {"IoU":>6s}']
for name, d in acc.items():
    res[name] = {}
    row = f'{name:12s} |'
    for key in ('all', 'right'):
        if key not in d: continue
        x = d[key]; P = x['tp'] / max(x['pred'], 1); R = x['tp'] / max(x['gt'], 1); I = x['tp'] / max(x['pred'] + x['gt'] - x['tp'], 1)
        res[name][key] = dict(precision=P, recall=R, iou=I, miou=float(np.mean(x['iou'])) if x['iou'] else 0.0)
        row += f' {P * 100:8.2f} {R * 100:6.2f} {I * 100:6.2f}' + (f' {res[name][key]["miou"] * 100:6.2f} |' if key == 'all' else '')
    lines.append(row)
txt = '\n'.join(lines); print(txt)
open(os.path.join(a.out, 'metrics.txt'), 'w').write(txt + '\n')
json.dump(dict(args=vars(a), results=res), open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
