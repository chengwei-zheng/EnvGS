"""LiDAR-derived mirror normal prior ("LiDAR gtnorm", 2026-10-20, plan: mirror_lidar/doc/MIRROR_SURFACE.md).

Builds <data_root>_lidarnorm/normals: StableNormal everywhere, except on mirror pixels found from the multi-echo LiDAR,
which get the normal of a surface fitted to the nearest LiDAR echoes; a band around the mirror boundary is stored black
(skipped with norm_ignore_zero=True). Uses only what a real TH-S02 also gives: per-zone echo ranges/directions
(th_s02_format/extracted/pointclouds_multiecho, ranges + xyz only), sensor poses, camera intrinsics/extrinsics.
Echo types, amplitudes and persp/masks are used for evaluation only (--eval).

    python lidar_mirror_normals.py data/datasets/mirror_b/r300 ~/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300 \
        --eval-mirror-normal 0.8660254 0.5 0

Steps: (1) candidate zones = >= 2 echoes, nearest echo by RANGE (synthetic echo_index is by strength; the ghost is
stronger than the mirror surface return); opening + connected components on the 192x256 zone grid (7x7, removes depth-edge
mixed pixels and see-through clutter), per-component planarity check. (2) multi-frame sequential RANSAC planes, split into in-plane connected
patches (1 cm grid); plane, or a local quadratic height field h(u,v) if it clearly beats the plane. (3) per camera:
ray-patch intersection on the front side, occluders from the same frame's LiDAR (nearest echo > 5 cm in front of the
surface), nearest-zone labelling, opening / small-component removal / erosion, eroded-away band -> black.
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
from scipy.spatial import cKDTree

# thresholds scale with the range noise sigma, estimated from the data (robust MAD of per-component plane residuals):
# synthetic TH-S02 surface returns have sigma ~2.5 cm, the real sensor ~1 cm
PLANE_K = 3.0          # plane inlier threshold = PLANE_K * sigma (>= 2 cm)
# see-through clutter / glass: the far echo is the real surface behind, ~100% on directly measured surfaces; a mirror's
# ghosts land there only by coincidence (standing mirror: reflected floor/ceiling stay on the floor/ceiling planes, ~50%)
SEE_THROUGH = 0.9
OPEN_ZONES = 7         # opening kernel on the 192x256 zone grid (min mirror width ~7 zones = 3.3 deg)
OCC_K = 4.0            # nearest echo more than OCC_K * sigma (>= 5 cm) in front of the surface -> occluder
GRID = 0.01            # in-plane occupancy grid (m)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_lidar(ds):
    ex = os.path.join(ds, 'th_s02_format', 'extracted')
    files = sorted(glob.glob(os.path.join(ex, 'pointclouds_multiecho', '*.npz')))
    poses = [l.split() for l in open(os.path.join(ex, 'gt_poses_tum.txt')) if l.strip() and not l.startswith('#')]
    assert len(poses) == len(files), (len(poses), len(files))
    T = []
    for p in poses:
        M = np.eye(4); M[:3, :3] = quat_to_R([float(v) for v in p[4:8]]); M[:3, 3] = [float(v) for v in p[1:4]]
        T.append(M)
    return files, T


def nearest_echo(f):
    d = np.load(f)
    r, xyz = d['ranges'].astype(np.float64), d['xyz'].astype(np.float64)    # (192,256,K), (192,256,K,3)
    valid = np.isfinite(r) & (r > 0)
    n = valid.sum(-1)
    rr = np.where(valid, r, np.inf)
    i = rr.argmin(-1)
    r1 = np.take_along_axis(rr, i[..., None], -1)[..., 0]
    p1 = np.take_along_axis(xyz, i[..., None, None].repeat(3, -1), 2)[:, :, 0]
    return n, r1, p1


def fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2]


def ransac_plane(P, th, iters=400, rng=None):
    rng = rng or np.random.default_rng(0)
    best, bn = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n)
        k = (np.abs((P - s[0]) @ n) < th).sum()
        if k > bn: bn, best = k, (s[0], n)
    c, n = best
    inl = np.abs((P - c) @ n) < th
    c, n = fit_plane(P[inl])
    return c, n, np.abs((P - c) @ n) < th


def frame_axes(n):
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, n); u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u), n], 1)     # columns u, v, n


class Patch:
    """Mirror patch: local frame (c, R = [u v n]), height field h(u,v) (plane: zero), occupancy grid in (u,v)."""

    def __init__(self, c, R, coef, grid, g0, info):
        self.c, self.R, self.coef, self.grid, self.g0, self.info = c, R, coef, grid, g0, info

    def h(self, u, v):
        a, b, cc, d, e, f = self.coef
        return a * u * u + b * u * v + cc * v * v + d * u + e * v + f

    def dh(self, u, v):
        a, b, cc, d, e, _ = self.coef
        return 2 * a * u + b * v + d, b * u + 2 * cc * v + e

    def intersect(self, o, d):
        """rays o (3,) or (N,3), d (N,3) world -> t (N,), inside mask (N,), local (u,v)."""
        ol = (np.broadcast_to(o, d.shape) - self.c) @ self.R
        dl = d @ self.R
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -ol[:, 2] / dl[:, 2]
            for _ in range(5):                                   # Newton on w(t) - h(u(t), v(t)) = 0 (no-op for a plane)
                p = ol + t[:, None] * dl
                hu, hv = self.dh(p[:, 0], p[:, 1])
                g = p[:, 2] - self.h(p[:, 0], p[:, 1])
                t = t - g / (dl[:, 2] - hu * dl[:, 0] - hv * dl[:, 1])
        p = ol + t[:, None] * dl
        ok = np.isfinite(t) & (t > 0)
        iu = np.floor((p[:, 0] - self.g0[0]) / GRID).astype(np.int64)
        iv = np.floor((p[:, 1] - self.g0[1]) / GRID).astype(np.int64)
        H, W = self.grid.shape
        ok &= (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        ok[ok] = self.grid[iv[ok], iu[ok]]
        return t, ok, p[:, 0], p[:, 1]

    def normal_world(self, u, v):
        hu, hv = self.dh(u, v)
        nl = np.stack([-hu, -hv, np.ones_like(hu)], -1)
        nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
        return nl @ self.R.T

    def front(self, o):
        ol = (o - self.c) @ self.R
        return ol[2] - self.h(ol[0], ol[1]) > 0


def candidate_points(files, T, train, min_zones):
    pts, fid, stats, sig = [], [], [], []
    # opening on the zone grid: mirrors are solid areas of multi-echo zones; see-through clutter (chair slats) gives
    # narrow / holey multi-echo regions. r300: 7x7 keeps 99.2% of mirror zones and 2.2% of the chair's (3x3: 38%)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_ZONES, OPEN_ZONES))
    for k in train:
        n, r1, p1 = nearest_echo(files[k])
        m = cv2.morphologyEx((n >= 2).astype(np.uint8), cv2.MORPH_OPEN, k3)
        nc, lab = cv2.connectedComponents(m, connectivity=4)
        kept = 0
        for c in range(1, nc):
            s = lab == c
            if s.sum() < min_zones: continue
            P = p1[s] @ T[k][:3, :3].T + T[k][:3, 3]
            cc, nn = fit_plane(P)
            for _ in range(2):                                    # refit on robust inliers
                e = (P - cc) @ nn
                sg = 1.4826 * np.median(np.abs(e - np.median(e)))
                inl = np.abs(e) < max(0.02, PLANE_K * sg)
                cc, nn = fit_plane(P[inl])
            if inl.mean() < 0.7 or sg > 0.04: continue           # not planar: depth-edge mixed pixels, clutter
            pts.append(P[inl]); fid.append(np.full(inl.sum(), k)); kept += inl.sum(); sig.append(sg)
        stats.append(((n >= 2).sum(), kept))
    return np.concatenate(pts), np.concatenate(fid), np.array(stats), float(np.median(sig))


def fit_patches(P, F, origins, sigma, min_pts=2000, min_frames=5, min_area=0.05):
    th = max(0.02, PLANE_K * sigma)
    patches, rest, rng = [], np.ones(len(P), bool), np.random.default_rng(0)
    for _ in range(6):
        idx = np.where(rest)[0]
        if len(idx) < min_pts: break
        sub = idx[rng.choice(len(idx), min(len(idx), 200000), replace=False)]
        c, n, _ = ransac_plane(P[sub], th, rng=rng)
        inl = idx[np.abs((P[idx] - c) @ n) < th]
        if len(inl) < min_pts: break
        rest[inl] = False
        R = frame_axes(n)
        L = (P[inl] - c) @ R
        g0 = L[:, :2].min(0) - 0.05
        W, H = (np.ceil((L[:, :2].max(0) + 0.05 - g0) / GRID)).astype(int)
        occ = np.zeros((H, W), np.uint8)
        occ[((L[:, 1] - g0[1]) / GRID).astype(int), ((L[:, 0] - g0[0]) / GRID).astype(int)] = 1
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        nc, lab = cv2.connectedComponents(occ, connectivity=8)
        iu, iv = ((L[:, 0] - g0[0]) / GRID).astype(int), ((L[:, 1] - g0[1]) / GRID).astype(int)
        for cc in range(1, nc):
            g = lab == cc
            sel = g[iv, iu]
            if g.sum() * GRID * GRID < min_area or sel.sum() < min_pts or len(np.unique(F[inl[sel]])) < min_frames: continue
            Q = P[inl[sel]]
            c2, n2 = fit_plane(Q)
            if np.mean((origins[F[inl[sel]]] - c2) @ n2) < 0: n2 = -n2      # normal toward the sensors that saw it
            R2 = frame_axes(n2)
            Lq = (Q - c2) @ R2
            rms_plane = np.sqrt(np.mean(Lq[:, 2] ** 2))
            u, v, w = Lq.T
            A = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], 1)
            coef = np.linalg.lstsq(A, w, rcond=None)[0]
            rms_quad = np.sqrt(np.mean((A @ coef - w) ** 2))
            quad = rms_plane > 0.012 and rms_quad < 0.8 * rms_plane
            if not quad: coef = np.zeros(6)
            g02 = Lq[:, :2].min(0) - 0.03
            W2, H2 = (np.ceil((Lq[:, :2].max(0) + 0.03 - g02) / GRID)).astype(int)
            # boundary at half the interior point density: range noise and the LiDAR footprint smear points past the
            # true edge, so raw occupancy overestimates the extent
            cnt = np.zeros((H2, W2), np.float32)
            np.add.at(cnt, (((Lq[:, 1] - g02[1]) / GRID).astype(int), ((Lq[:, 0] - g02[0]) / GRID).astype(int)), 1)
            den = cv2.GaussianBlur(cnt, (0, 0), 2.0)
            core = den > np.percentile(den[den > 0], 50)
            occ2 = (den > 0.5 * np.median(den[core])).astype(np.uint8)
            occ2 = cv2.morphologyEx(occ2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            nc2, lab2 = cv2.connectedComponents(occ2, connectivity=8)
            big = 1 + np.argmax(np.bincount(lab2.ravel())[1:])
            # leftovers of an already-found mirror (its noise tails can still form a tilted "plane" through it)
            if any(q.intersect(c2 + q.R[:, 2] * 0.1, -q.R[:, 2][None])[1][0] and abs((c2 - q.c) @ q.R[:, 2]) < 0.1 for q in patches):
                continue
            if rms_plane > 1.3 * max(0.01, sigma):                       # not a clean plane at the measured noise level
                continue
            info = dict(n_pts=int(len(Q)), n_frames=int(len(np.unique(F[inl[sel]]))), area_m2=float((lab2 == big).sum() * GRID * GRID),
                        rms_plane_mm=1000 * rms_plane, rms_quad_mm=1000 * rms_quad, model='quadratic' if quad else 'plane',
                        center=c2.tolist(), normal=n2.tolist())
            patches.append(Patch(c2, R2, coef, lab2 == big, g02, info))
    return patches


def far_echo(f):
    d = np.load(f)
    r, xyz = d['ranges'].astype(np.float64), d['xyz'].astype(np.float64)
    rr = np.where(np.isfinite(r) & (r > 0), r, -np.inf)
    i = rr.argmax(-1)
    return np.take_along_axis(xyz, i[..., None, None].repeat(3, -1), 2)[:, :, 0]


def direct_voxels(files, T, frames, vox=0.05):
    """Voxel set of surfaces measured directly: nearest echo of single-echo zones, all given frames."""
    keys = []
    for k in frames:
        n, _, p1 = nearest_echo(files[k])
        P = p1[n == 1] @ T[k][:3, :3].T + T[k][:3, 3]
        keys.append(np.floor(P / vox).astype(np.int64))
    return set(map(tuple, np.unique(np.concatenate(keys), axis=0))), vox


def seen_through_ratio(p, files, T, D, frames, sigma, dvox, n_frames=40):
    """Fraction of the patch's far echoes that land on a directly measured surface (within one voxel): see-through
    clutter / glass -> high (the far echo is the real thing behind it); mirror -> low (the far echo is a ghost)."""
    vs, vox = dvox
    nb = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing='ij'), -1).reshape(-1, 3)
    dz = D.reshape(-1, 3); okz = np.isfinite(dz[:, 0]); tot = hit = 0
    for k in frames[::max(1, len(frames) // n_frames)]:
        o = T[k][:3, 3]
        if not p.front(o): continue
        n, r1, _ = nearest_echo(files[k])
        tz, iz, _, _ = p.intersect(o, dz[okz] @ T[k][:3, :3].T)
        s = iz & (np.abs(r1.reshape(-1)[okz] - tz) < max(0.02, PLANE_K * sigma)) & (n.reshape(-1)[okz] >= 2)
        if not s.any(): continue
        Pf = far_echo(files[k]).reshape(-1, 3)[okz][s] @ T[k][:3, :3].T + T[k][:3, 3]
        kk = np.floor(Pf / vox).astype(np.int64)
        tot += len(kk); hit += sum(any(tuple(q + d) in vs for d in nb) for q in kk[::max(1, len(kk) // 300)]) * max(1, len(kk) // 300)
    return hit / max(1, tot)


def coverage_ok(p, files, T, D, train, sigma, n_frames=40, min_cov=0.7):
    """Within the patch's convex hull, of the LiDAR zones whose NEAREST echo lies on the surface, the fraction that also
    have a second echo. A mirror surface return comes with its ghost (~1 synthetic, ~0.9 real); planar see-through
    clutter (chair slats, shelves) is mostly single-echo where the beam fully hits the slats."""
    grid = p.grid
    cs, _ = cv2.findContours(grid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(grid, np.uint8); cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cs))], 1)
    p.grid = hull.astype(bool)
    dz = D.reshape(-1, 3); okz = np.isfinite(dz[:, 0]); on = multi = 0
    for k in train[::max(1, len(train) // n_frames)]:
        o = T[k][:3, 3]
        if not p.front(o): continue
        n, r1, _ = nearest_echo(files[k])
        tz, iz, _, _ = p.intersect(o, dz[okz] @ T[k][:3, :3].T)
        s = iz & (np.abs(r1.reshape(-1)[okz] - tz) < max(0.02, PLANE_K * sigma))
        on += s.sum(); multi += (s & (n.reshape(-1)[okz] >= 2)).sum()
    p.grid = grid
    cov = multi / max(1, on)
    p.info['multi_echo_ratio_on_surface'] = float(cov)
    print(f"patch at {np.round(p.c, 2).tolist()} area {p.info['area_m2']:.2f} m2: {on} zones with nearest echo on the surface "
          f"(convex hull), {cov:.2f} of them multi-echo -> {'mirror' if cov >= min_cov else 'rejected'}")
    return cov >= min_cov


def zone_dirs(files, step=5):
    acc = np.zeros((192, 256, 3)); cnt = np.zeros((192, 256))
    for f in files[::step]:
        d = np.load(f); r, xyz = d['ranges'], d['xyz']
        for e in range(r.shape[-1]):
            v = np.isfinite(r[..., e]) & (r[..., e] > 0)
            acc[v] += xyz[..., e, :][v] / r[..., e][v][:, None]; cnt[v] += 1
    D = acc / np.maximum(cnt[..., None], 1)
    D /= np.maximum(np.linalg.norm(D, axis=-1, keepdims=True), 1e-9)
    D[cnt == 0] = np.nan
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root')
    ap.add_argument('blender_ds')
    ap.add_argument('--suffix', default='lidarnorm')
    ap.add_argument('--min-zones', type=int, default=30)
    ap.add_argument('--band', type=float, default=1.0, help='boundary band half-width, in LiDAR zone spacings')
    ap.add_argument('--eval-mirror-normal', type=float, nargs=3, default=None, help='GT mirror normal, evaluation only')
    ap.add_argument('--preview', type=int, nargs='*', default=[0, 40, 88, 136, 200, 264])
    ap.add_argument('--only', type=int, nargs='*', default=None, help='process only these frames (quick test)')
    a = ap.parse_args()
    src = os.path.abspath(a.data_root.rstrip('/'))
    out = src + '_' + a.suffix
    dbg = out + '_debug'
    os.makedirs(os.path.join(out, 'normals'), exist_ok=True); os.makedirs(dbg, exist_ok=True)
    os.makedirs(os.path.join(out, 'envs'), exist_ok=True)
    for f in ('images', 'sparse', 'intri.yml', 'extri.yml'):
        if not os.path.lexists(os.path.join(out, f)): os.symlink(os.path.join(src, f), os.path.join(out, f))
    if os.path.exists(os.path.join(src, 'envs', 'points3D.ply')):
        shutil.copy2(os.path.join(src, 'envs', 'points3D.ply'), os.path.join(out, 'envs', 'points3D.ply'))

    names = sorted(os.listdir(os.path.join(src, 'normals')))
    fe, fi = cv2.FileStorage(os.path.join(src, 'extri.yml'), 0), cv2.FileStorage(os.path.join(src, 'intri.yml'), 0)
    Rc = {n: fe.getNode(f'Rot_{n}').mat() for n in names}
    Tc = {n: fe.getNode(f'T_{n}').mat().ravel() for n in names}
    Kc = {n: fi.getNode(f'K_{n}').mat() for n in names}
    files, T = load_lidar(a.blender_ds)
    assert len(files) == len(names), (len(files), len(names))
    origins = np.array([t[:3, 3] for t in T])
    train = [k for k in range(len(files)) if k % 8 != 0]          # EnvGS split: every 8th frame is val

    P, F, st, sigma = candidate_points(files, T, train, a.min_zones)
    print(f'candidates: {len(P)} points from {len(np.unique(F))} train frames '
          f'(multi-echo zones {st[:, 0].sum()}, kept after opening/components/planarity {st[:, 1].sum()}); '
          f'range noise sigma (MAD) {100 * sigma:.2f} cm')
    patches = fit_patches(P, F, origins, sigma)
    for i, p in enumerate(patches): print(f'patch {i}:', json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 2) if isinstance(v, float) else v)) for k, v in p.info.items()}))
    D = zone_dirs(files)
    patches = [p for p in patches if coverage_ok(p, files, T, D, train, sigma)]
    dvox = direct_voxels(files, T, train[::3])
    keep = []
    for p in patches:
        st_ = seen_through_ratio(p, files, T, D, train, sigma, dvox)
        p.info['far_echo_on_direct_surface'] = float(st_)
        print(f"patch at {np.round(p.c, 2).tolist()}: {st_:.2f} of far echoes land on directly measured surfaces "
              f"-> {'rejected (see-through)' if st_ > SEE_THROUGH else 'mirror'}")
        if st_ <= SEE_THROUGH: keep.append(p)
    patches = keep
    assert patches, 'no mirror patch found'
    spacing = np.radians(120 / 256)

    if a.only is not None: names = [n for n in names if int(n) in a.only]
    gt_n = None
    if a.eval_mirror_normal is not None:
        gt_n = np.array(a.eval_mirror_normal, float); gt_n /= np.linalg.norm(gt_n)
        for i, p in enumerate(patches):
            print(f'patch {i} vs GT: normal angle {np.degrees(np.arccos(abs(p.R[:, 2] @ gt_n))):.3f} deg')
    tf = json.load(open(os.path.join(a.blender_ds, 'persp', 'transforms.json')))
    ev = []
    for n in names:
        k = int(n)
        nm = cv2.imread(os.path.join(src, 'normals', n, '000000.png'), cv2.IMREAD_UNCHANGED)
        H, W = nm.shape[:2]; scale = 65535.0 if nm.dtype == np.uint16 else 255.0
        R, t, K = Rc[n], Tc[n], Kc[n]
        o = -R.T @ t
        px = int(np.ceil(K[0, 0] * np.tan(spacing)))                               # zone spacing in pixels (image centre)
        ys, xs = np.mgrid[0:H, 0:W]
        dc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1).reshape(-1, 3)
        dw = dc @ R; dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        mir = np.zeros(H * W, bool); unc = np.zeros(H * W, bool); nrm = np.zeros((H * W, 3)); tbest = np.full(H * W, np.inf)
        # same-frame LiDAR: nearest echo per zone, labelled occluder / mirror where its ray hits a patch
        _, r1, _ = nearest_echo(files[k])
        dz = D.reshape(-1, 3); okz = np.isfinite(dz[:, 0])
        dzw = dz[okz] @ T[k][:3, :3].T; os_ = T[k][:3, 3]; r1z = r1.reshape(-1)[okz]
        for p in patches:
            if not p.front(o): continue                                               # camera behind the mirror: mirror back
            tp, inside, u, v = p.intersect(o, dw)
            cand = inside & (tp < tbest)                                              # nearest patch wins where patches overlap
            tbest[cand] = tp[cand]; mir[cand] = False; unc[cand] = False
            if not cand.any(): continue
            tz, iz, _, _ = p.intersect(os_, dzw)
            occ = iz & (r1z < tz - max(0.05, OCC_K * sigma))                           # inf r1 (no echo) -> mirror
            og = np.zeros(okz.shape, np.float32); og[np.where(okz)[0][occ]] = 1
            og = og.reshape(192, 256)
            og = og * (cv2.boxFilter(og, -1, (3, 3), normalize=False) >= 3)                 # drop isolated occluder zones (noise)
            occ = occ & (og.reshape(-1)[okz] > 0)
            Xz = np.where(occ[:, None], os_ + np.where(np.isfinite(r1z), r1z, 0)[:, None] * dzw, os_ + tz[:, None] * dzw)[iz]
            lab = occ[iz]
            xc = Xz @ R.T + t; uv = xc[:, :2] / xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]] - 0.5
            okp = xc[:, 2] > 0
            ci = np.where(cand)[0]
            if okp.sum() == 0: unc[ci] = True; continue
            dist, j = cKDTree(uv[okp]).query(np.stack([xs.ravel()[ci], ys.ravel()[ci]], 1), workers=8)
            far = dist > 1.5 * px
            is_m = ~lab[okp][j] & ~far
            unc[ci[far]] = True
            mir[ci[is_m]] = True
            nrm[ci[is_m]] = p.normal_world(u[ci[is_m]], v[ci[is_m]])
        mir = mir.reshape(H, W).astype(np.uint8); unc = unc.reshape(H, W)
        mir = cv2.morphologyEx(mir, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        nc, lab, sts, _ = cv2.connectedComponentsWithStats(mir, connectivity=8)
        for c in range(1, nc):
            if sts[c, cv2.CC_STAT_AREA] < 0.001 * H * W: mir[lab == c] = 0
        b = max(1, int(round(a.band * px)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
        inner = cv2.erode(mir, ker).astype(bool)
        band = (cv2.dilate(mir, ker).astype(bool) & ~inner) | (unc & ~inner)
        res = nm.copy()
        ncam = nrm.reshape(H, W, 3)[inner] @ R.T
        res[inner] = np.round((ncam + 1) / 2 * scale)[:, ::-1].astype(nm.dtype)
        res[band] = 0
        os.makedirs(os.path.join(out, 'normals', n), exist_ok=True)
        cv2.imwrite(os.path.join(out, 'normals', n, '000000.png'), res)
        if gt_n is not None:
            g = cv2.imread(os.path.join(a.blender_ds, 'persp', 'masks', f'mask_{k:04d}.png'), cv2.IMREAD_UNCHANGED) == 3
            gn = R @ gt_n
            ang = np.degrees(np.arccos(np.clip(np.abs(ncam @ gn), -1, 1))) if inner.any() else np.zeros(0)
            ev.append(dict(frame=k, gt=int(g.sum()), inner=int(inner.sum()), tp=int((inner & g).sum()), band=int(band.sum()),
                           band_gt=int((band & g).sum()), iou=float((mir.astype(bool) & g).sum() / max(1, (mir.astype(bool) | g).sum())),
                           ang_med=float(np.median(ang)) if len(ang) else np.nan, ang_p99=float(np.percentile(ang, 99)) if len(ang) else np.nan))
            if k in a.preview:
                rgb = cv2.imread(os.path.join(src, 'images', n, '000000.png'))
                ov = rgb.copy()
                ov[inner & g] = (0.5 * ov[inner & g] + [0, 127, 0]).astype(np.uint8)          # painted, correct: green
                ov[inner & ~g] = (0.5 * ov[inner & ~g] + [127, 0, 0]).astype(np.uint8)        # painted, not mirror: blue
                ov[g & ~inner & ~band] = (0.5 * ov[g & ~inner & ~band] + [0, 0, 127]).astype(np.uint8)  # missed: red
                ov[band] = (0.5 * ov[band] + [0, 127, 127]).astype(np.uint8)                  # ignored band: yellow
                row = np.concatenate([cv2.resize(x, (648, 486), interpolation=cv2.INTER_AREA) for x in (rgb, ov, nm, res)], 1)
                cv2.putText(row, f'frame {k}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(dbg, f'frame{k:04d}.jpg'), row, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ev:
        e = [x for x in ev if x['gt'] > 0 or x['inner'] > 0]
        tp, inn, gt = sum(x['tp'] for x in e), sum(x['inner'] for x in e), sum(x['gt'] for x in e)
        print(f'eval over {len(e)} frames with mirror (GT or painted): precision {tp / max(1, inn):.4f}, recall(painted/GT) {tp / max(1, gt):.3f}, '
              f'GT mirror px in ignored band {sum(x["band_gt"] for x in e) / max(1, gt):.3f}, missed {1 - (tp + sum(x["band_gt"] for x in e)) / max(1, gt):.3f}, '
              f'mask IoU median {np.median([x["iou"] for x in e]):.3f}, painted normal error median {np.nanmedian([x["ang_med"] for x in e]):.3f} deg '
              f'(worst frame p99 {np.nanmax([x["ang_p99"] for x in e]):.2f})')
        json.dump(dict(patches=[p.info for p in patches], frames=ev), open(os.path.join(dbg, 'eval.json'), 'w'), indent=1)
        rows = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(dbg, 'frame*.jpg')))]
        if rows: cv2.imwrite(os.path.join(dbg, 'preview_sheet.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('->', out, '| debug', dbg)


if __name__ == '__main__':
    main()
