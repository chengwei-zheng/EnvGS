"""Visual checks of the mirror preprocessing (2026-11-07, DATA_PREP.md steps 4 and 5): videos over the RGB training images, every
--step-th camera.

  surface  the explicit mirror surface as msurf builds it at initialisation (mirror_surface.py, from the fit's patch<i>.npz):
           every pixel ray is intersected with the surface (no occlusion test: objects in front of the mirror are painted over);
           hit pixels inside the extent are tinted with the world normal ((n + 1) / 2), white lines every --grid m of the patch
           (u, v) coordinates, yellow = extent boundary, cyan = the rim-drop line (the outer --rim-drop strip was left out of
           the fit and is extrapolated), magenta = the undilated extent when --extent-dilate > 0.
  mask     the per-camera mirror masks (lidar_mirror_mask.py / gt_mirror_masks.py format): 255 interior green, 128 boundary
           band yellow; --gt <blender persp dir> adds the GT mirror outline (persp/masks == 3) in magenta (evaluation only).

    python mirror_lidar/tools/vis_mirror_prep.py surface <data_root> <patch0.npz> <out.mp4> [--type spline] [--extent-dilate 0.04]
    python mirror_lidar/tools/vis_mirror_prep.py mask <data_root> <mirror_masks dir> <out.mp4> [--gt <persp dir>]
common: --step 4 (every 4th camera), --fps 8, --scale 0.5 (of the full image resolution), --only-mirror (skip frames without
mirror pixels), --gpu. Camera i <-> images/<name i> (sorted), intri.yml / extri.yml of <data_root> (EasyVolcap, OpenCV, z-depth).
"""
import argparse, os, sys
import cv2
import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv   # EasyVolcap parses sys.argv at import time
import torch
from easyvolcap.models.samplers.mirror_surface import build_mirror_surface
sys.argv = _argv


def cameras(data_root):
    names = sorted(os.listdir(os.path.join(data_root, 'images')))
    fe, fi = cv2.FileStorage(os.path.join(data_root, 'extri.yml'), 0), cv2.FileStorage(os.path.join(data_root, 'intri.yml'), 0)
    return [(n, fe.getNode(f'Rot_{n}').mat(), fe.getNode(f'T_{n}').mat().ravel(), fi.getNode(f'K_{n}').mat()) for n in names]


def image(data_root, n, W, H):
    return cv2.resize(cv2.imread(os.path.join(data_root, 'images', n, '000000.png')), (W, H), interpolation=cv2.INTER_AREA)


def edge(m, px=1):
    """boundary pixels of a boolean mask (inner side), px thick"""
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return m & ~cv2.erode(m.astype(np.uint8), k).astype(bool)


def level_lines(f, step, valid):
    """pixels where floor(f / step) changes to a neighbour (iso-lines of f every step), inside valid"""
    q = np.floor(f / step)
    e = np.zeros_like(valid)
    e[:, 1:] |= (q[:, 1:] != q[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    e[1:, :] |= (q[1:, :] != q[:-1, :]) & valid[1:, :] & valid[:-1, :]
    return e


def label(img, lines):
    for i, s in enumerate(lines):
        cv2.putText(img, s, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)


class Video:
    def __init__(self, path, W, H, fps):
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.w = imageio_ffmpeg.write_frames(path, (W, H), fps=fps, quality=8, macro_block_size=2)
        self.w.send(None)
        self.n = 0

    def add(self, bgr):
        self.w.send(np.ascontiguousarray(bgr[..., ::-1])); self.n += 1

    def close(self):
        self.w.close()


@torch.no_grad()
def surface_video(a, cams, W, H, W0, H0, dev):
    surf = build_mirror_surface(type=a.type, fit=a.fit, extent_dilate=a.extent_dilate).to(dev).eval()
    from scipy import ndimage
    fit = np.load(a.fit)
    e0 = fit['extent'].astype(bool)
    ext0 = torch.as_tensor(e0, device=dev)
    rim_drop = float(fit['rim_drop'])
    rd = torch.as_tensor(ndimage.distance_transform_edt(e0) * float(fit['cell']), dtype=torch.float, device=dev)  # to the fit's extent boundary (m)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing='ij')
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)
    vid = Video(a.out, W, H, a.fps)
    for k in range(0, len(cams), a.step):
        n, R, t, K = cams[k]
        K = K.copy(); K[0] *= W / W0; K[1] *= H / H0
        Rt, Kt, tt = (torch.as_tensor(x, dtype=torch.float, device=dev) for x in (R, K, t))
        d = pix @ torch.linalg.inv(Kt).T @ Rt                   # world directions, camera z = 1
        o = (-Rt.T @ tt).expand_as(d)
        ol, dl = (o - surf.c) @ surf.R, d @ surf.R
        tr, ok = surf.intersect_local(ol, dl)
        x = ol + tr[:, None] * dl
        hit = ok & (tr > 0) & surf.inside(x)
        if a.only_mirror and not hit.any(): continue
        nw = torch.nn.functional.normalize(surf.normal_local(x) @ surf.R.T, dim=-1)
        nw = torch.where((nw * d).sum(-1, keepdim=True) > 0, -nw, nw)
        ij = torch.floor((x[:, :2] - surf.g0) / surf.cell).long()
        ij[:, 0].clamp_(0, rd.shape[1] - 1); ij[:, 1].clamp_(0, rd.shape[0] - 1)
        rdist = rd[ij[:, 1], ij[:, 0]]
        in0 = ext0[ij[:, 1], ij[:, 0]] & hit
        hit_, uv, rdist, in0, nw = (v.reshape(H, W, *v.shape[1:]).cpu().numpy() for v in (hit, x[:, :2], rdist, in0, nw))
        img = image(a.data_root, n, W, H).astype(np.float32)
        col = ((nw + 1) / 2 * 255)[..., ::-1]                   # RGB normal -> BGR
        img[hit_] = (1 - a.alpha) * img[hit_] + a.alpha * col[hit_]
        img = img.astype(np.uint8)
        img[level_lines(uv[..., 0], a.grid, hit_) | level_lines(uv[..., 1], a.grid, hit_)] = (255, 255, 255)
        img[level_lines(np.where(hit_, rdist, 0), rim_drop, hit_) & (np.abs(rdist - rim_drop) < 0.02)] = (255, 255, 0)
        if a.extent_dilate > 0: img[edge(in0, 1)] = (255, 0, 255)
        img[edge(hit_, 1)] = (0, 255, 255)
        label(img, [f'cam {n} ({"val" if k % 8 == 0 else "train"})  surface {hit_.mean() * 100:.1f}% of image',
                    f'{os.path.basename(os.path.dirname(a.fit))}/{os.path.basename(a.fit)} {a.type}'
                    + (f' extent +{a.extent_dilate * 100:.0f} cm' if a.extent_dilate > 0 else '')])
        vid.add(img)
    vid.close()
    return vid.n


def mask_video(a, cams, W, H):
    gt_files = None
    if a.gt:
        gt_files = sorted(f for f in os.listdir(os.path.join(a.gt, 'masks')) if f.endswith('.png'))
        assert len(gt_files) == len(cams), (len(gt_files), len(cams))
    vid = Video(a.out, W, H, a.fps)
    for k in range(0, len(cams), a.step):
        n = cams[k][0]
        m = cv2.imread(os.path.join(a.masks, n, '000000.png'), cv2.IMREAD_UNCHANGED)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        g = None
        if gt_files:
            g = cv2.resize(cv2.imread(os.path.join(a.gt, 'masks', gt_files[k]), cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_NEAREST)
            g = (g[..., 0] if g.ndim == 3 else g) == 3
        if a.only_mirror and not (m > 0).any() and (g is None or not g.any()): continue
        img = image(a.data_root, n, W, H).astype(np.float32)
        for val, col in ((255, (0, 255, 0)), (128, (0, 255, 255))):
            s = m == val
            img[s] = (1 - a.alpha) * img[s] + a.alpha * np.array(col, np.float32)
        img = img.astype(np.uint8)
        img[edge(m == 255, 1)] = (0, 255, 0)
        lines = [f'cam {n} ({"val" if k % 8 == 0 else "train"})  interior {(m == 255).mean() * 100:.1f}%  band {(m == 128).mean() * 100:.1f}%',
                 f'{os.path.basename(os.path.dirname(os.path.normpath(a.masks)))}/{os.path.basename(os.path.normpath(a.masks))}']
        if g is not None:
            img[edge(g, 2)] = (255, 0, 255)
            if g.any():
                lines.append(f'GT (magenta, eval only): interior precision {((m == 255) & g).sum() / max((m == 255).sum(), 1) * 100:.1f}%  '
                             f'recall {((m == 255) & g).sum() / g.sum() * 100:.1f}%')
        label(img, lines)
        vid.add(img)
    vid.close()
    return vid.n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('surface'); s.add_argument('data_root'); s.add_argument('fit'); s.add_argument('out')
    s.add_argument('--type', default='spline', choices=['spline', 'quadric'])
    s.add_argument('--extent-dilate', type=float, default=0.0, help='(m), as mirror_surface_cfg.extent_dilate')
    s.add_argument('--grid', type=float, default=0.1, help='(u, v) grid line spacing (m)')
    s.add_argument('--alpha', type=float, default=0.45)
    s = sub.add_parser('mask'); s.add_argument('data_root'); s.add_argument('masks'); s.add_argument('out')
    s.add_argument('--gt', default='', help='blender persp dir: GT mirror outline (evaluation only)')
    s.add_argument('--alpha', type=float, default=0.4)
    for s in sub.choices.values():
        s.add_argument('--step', type=int, default=4, help='every step-th camera')
        s.add_argument('--fps', type=int, default=8)
        s.add_argument('--scale', type=float, default=0.5, help='of the full image resolution')
        s.add_argument('--only-mirror', action='store_true', help='skip frames without mirror pixels')
        s.add_argument('--gpu', default='0')
    a = ap.parse_args()
    cams = cameras(a.data_root)
    H0, W0 = cv2.imread(os.path.join(a.data_root, 'images', cams[0][0], '000000.png')).shape[:2]
    W, H = int(round(W0 * a.scale / 2)) * 2, int(round(H0 * a.scale / 2)) * 2
    if a.mode == 'surface':
        dev = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'
        nf = surface_video(a, cams, W, H, W0, H0, dev)
    else:
        nf = mask_video(a, cams, W, H)
    print(f'{a.mode}: {nf} frames ({W}x{H}, every {a.step}th camera, {a.fps} fps) -> {a.out}')
