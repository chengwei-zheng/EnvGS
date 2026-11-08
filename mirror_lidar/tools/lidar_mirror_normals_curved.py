"""LiDAR-derived normal prior for CURVED (or planar) mirrors (2026-10-29, corridor convex-mirror scene, dataset_c480).

Same output as lidar_mirror_normals.py (<data_root>_lidarnorm/normals: StableNormal everywhere, the fitted mirror normal
on mirror pixels, black boundary band) and the same mirror evidence (multi-echo zones, surface multi-echo coverage,
see-through check, same-frame occluders), but no planar assumption anywhere:
  (1) candidate components: per-frame multi-echo zones after the 7x7 opening; the planarity check is replaced by a
      smoothness check (robust degree-4 height field in the component's PCA frame, MAD < 4 cm, >= 70% inliers);
  (2) patches: 3 cm voxel connected components of all candidate points (instead of sequential RANSAC planes);
  (3) surface: per patch, points averaged on 1 cm cells of the patch's (u, v) grid (robust per-cell mean, weights =
      counts: every echo counts, and the per-cell averaging brings the normal-direction noise from ~2.5 cm down to a few
      mm, which removes the noise-induced curvature bias of algebraic fits), PCA normals, then the free-form quadric MLS of
      quadric_mls.py (plane / sphere reproduced exactly, kappa1 - kappa2 prior tau) evaluated on a 1 cm (u, v) grid
      in the patch frame -> height field + normal field; rays intersect it by Newton on the bilinear height field.
      --surface quadric (2026-11-02, cv300): instead of the local MLS, ONE implicit quadric (10 coefficients, Taubin fit,
      Tukey IRLS on the geometric distance) over all cells at least --rim-drop inside the extent, then sampled on the same
      grid. The h = 0.2 m MLS extrapolated a flattened surface over the outer ~10 cm of the 0.54 m cv300 mirror (one-sided
      kernel): 28% of the mirror pixels at 1.3-2.8 deg, which made the reflection worse than the 2DGS normals.
      Planes and spheres are reproduced exactly, and a global model extrapolates them correctly to the rim.
  (4) per camera: identical to lidar_mirror_normals.py.

    python lidar_mirror_normals_curved.py data/datasets/corridor/c480 ~/SMBC/blender/project/proj_indoor_corridor_convex/dataset_c480 \
        --eval-sphere -0.02669 2.42669 2.25706 1.0

--eval-sphere cx cy cz R (convex, normal = (x - c) / R) or --eval-mirror-normal nx ny nz (plane): evaluation only.
--fit-only: fit + surface evaluation (+ --export-ply), no per-camera normals / masks (nothing written to <data_root>_<suffix>).
--export-ply DIR: per patch, the fitted surface as a triangle mesh over its extent (patch<i>_surface.ply, with vertex
normals), the 1 cm cell means used by the fit (patch<i>_cells.ply, colour = distance to the extent boundary) and patch<i>.npz
(frame, extent, height field, quadric, cell means: the input of the trainable mirror surface, mirror_surface.py).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_mirror_normals as lmn  # noqa: E402
import quadric_mls as qm  # noqa: E402

GRID = lmn.GRID            # 1 cm (u, v) grid of the height field (lmn.Patch.intersect uses the same constant)
VOX_CLUSTER = 0.03         # voxel size of the patch clustering
SMOOTH_MAD = 0.04          # candidate component: robust residual of the degree-4 height field (m)
H_MLS = 0.20               # MLS kernel radius (m), on the cell means (c480 sweep 10-29: best of 0.1 / 0.15 / 0.2)
TAU = 0.3                  # kappa1 - kappa2 prior (1/m) (c480: 0.3 beats 1.0 at the rim, 1.08 vs 1.76 deg median)
RIM_DROP = 0.03            # outer ring of the extent left out of the fit (m)


def frame_pca(P, origins_mean):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] if (origins_mean - c) @ vt[2] > 0 else -vt[2]
    return c, lmn.frame_axes(n)


def poly4(u, v):
    return np.stack([u ** i * v ** j for i in range(5) for j in range(5 - i)], 1)


def fit_height(L, iters=3, k=3.0):
    """robust degree-4 height field w(u, v) in a local frame; returns coef, residual MAD, inlier mask."""
    A = poly4(L[:, 0], L[:, 1])
    inl = np.ones(len(L), bool)
    for _ in range(iters):
        coef = np.linalg.lstsq(A[inl], L[inl, 2], rcond=None)[0]
        e = L[:, 2] - A @ coef
        sg = 1.4826 * np.median(np.abs(e - np.median(e)))
        inl = np.abs(e) < max(0.02, k * sg)
    return coef, sg, inl


def quadric_terms(X):
    """monomials of the implicit quadric and their gradients: v (N, 10), dv/dx, dv/dy, dv/dz (N, 10 each)."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    o, l = np.zeros_like(x), np.ones_like(x)
    v = np.stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, l], 1)
    gx = np.stack([2 * x, o, o, y, z, o, l, o, o, o], 1)
    gy = np.stack([o, 2 * y, o, x, o, z, o, l, o, o], 1)
    gz = np.stack([o, o, 2 * z, o, x, y, o, o, l, o], 1)
    return v, gx, gy, gz


def fit_quadric(X, w, iters=8, k=4.0):
    """weighted Taubin fit of an implicit quadric f(x) = q . v(x) = 0 to local points X (N, 3), Tukey IRLS on the first-order
    geometric distance |f| / |grad f|. Returns q (in the coordinates of X), robust residual scale (m), final weights."""
    from scipy.linalg import eig
    v, gx, gy, gz = quadric_terms(X)
    ww = w.copy()
    for it in range(iters):
        M = (v * ww[:, None]).T @ v
        Nm = sum((g * ww[:, None]).T @ g for g in (gx, gy, gz))
        ev, V = eig(M, Nm)
        ev = np.real(ev); ok = np.isfinite(ev) & (ev > -1e-12)
        q = np.real(V[:, np.where(ok)[0][np.argmin(ev[ok])]])
        g = np.sqrt((gx @ q) ** 2 + (gy @ q) ** 2 + (gz @ q) ** 2) + 1e-12
        r = (v @ q) / g
        sg = 1.4826 * np.median(np.abs(r)) + 1e-5
        u = r / (k * sg)
        ww = w * np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0)
    return q / np.linalg.norm(q), sg, ww


def quadric_height(q, u, v, z0, branch_mask=None, par_tol=None):
    """height z(u, v) on the implicit quadric (root closest to the guess z0) and the slopes dz/du, dz/dv.
    branch_mask (2026-11-04): pick ONE root branch for all samples (the one closest to z0 in median over branch_mask) instead of
    per sample. A flat mirror makes the algebraic fit degenerate into two parallel planes (f300: z^2 - 8e-5 z - 1e-5 = 0, 3.6 mm
    apart); the per-sample choice jumped between them (3.6 mm steps in Hh -> 4-12 deg slopes in the B-spline init of mirror_surface.py,
    dead-beta patches in envgs_f300_msurf_spline_learn_ldepth005_dz5). par_tol: if the two branches stay within par_tol of each
    other over branch_mask (parallel sheets), use their mid-surface z = -b / 2a (the implicit gradient vanishes there, so its slopes
    are the derivatives of -b / 2a)."""
    a = q[2] * np.ones_like(u)
    b = q[4] * u + q[5] * v + q[8]
    c = q[0] * u * u + q[1] * v * v + q[3] * u * v + q[6] * u + q[7] * v + q[9]
    with np.errstate(divide='ignore', invalid='ignore'):
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        lin = np.abs(a) < 1e-9 * (np.abs(b) + 1e-12)
        r1 = np.where(lin, -c / b, (-b + sq) / (2 * a))
        r2 = np.where(lin, -c / b, (-b - sq) / (2 * a))
    if branch_mask is None:
        z = np.where(np.abs(r1 - z0) < np.abs(r2 - z0), r1, r2)
    else:
        bm = branch_mask & np.isfinite(r1) & np.isfinite(r2) & (disc >= 0)
        gap = np.abs(r1 - r2)[bm].max()
        if par_tol is not None and not lin.any() and gap < par_tol:
            print(f'quadric_height: degenerate fit, two parallel sheets <= {gap:.4g} apart (fit units) -> mid-surface')
            z = -b / (2 * a)
            return z, -q[4] / (2 * a), -q[5] / (2 * a), np.isfinite(z)
        z = r1 if np.median(np.abs(r1 - z0)[bm]) < np.median(np.abs(r2 - z0)[bm]) else r2
    ok = np.isfinite(z) & ((disc >= 0) | lin)
    X = np.stack([u, v, np.where(ok, z, 0)], 1)
    _, gx, gy, gz = quadric_terms(X)
    fx, fy, fz = gx @ q, gy @ q, gz @ q
    with np.errstate(divide='ignore', invalid='ignore'):
        hu, hv = -fx / fz, -fy / fz
    ok &= np.isfinite(hu) & np.isfinite(hv) & (np.abs(fz) > 0.2 * np.sqrt(fx * fx + fy * fy + fz * fz))
    return z, hu, hv, ok


def write_ply(path, V, F=None, N=None, C=None):
    """binary little-endian PLY (MeshLab): vertices V (n, 3), optional faces F (m, 3), normals N, colours C (uint8)."""
    V = np.asarray(V, np.float32)
    props = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    cols = [V]
    if N is not None: props += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]; cols.append(np.asarray(N, np.float32))
    vert = np.zeros(len(V), dtype=props + ([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')] if C is not None else []))
    vert['x'], vert['y'], vert['z'] = V.T
    if N is not None: vert['nx'], vert['ny'], vert['nz'] = np.asarray(N, np.float32).T
    if C is not None: vert['red'], vert['green'], vert['blue'] = np.asarray(C, np.uint8).T
    with open(path, 'wb') as f:
        hdr = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n'
        hdr += ''.join(f'property {"float" if np.dtype(t).kind == "f" else "uchar"} {n}\n' for n, t in vert.dtype.descr)
        if F is not None: hdr += f'element face {len(F)}\nproperty list uchar int vertex_indices\n'
        f.write((hdr + 'end_header\n').encode())
        f.write(vert.tobytes())
        if F is not None:
            fa = np.zeros(len(F), dtype=[('n', 'u1'), ('i', '<i4', (3,))]); fa['n'] = 3; fa['i'] = F
            f.write(fa.tobytes())


def patch_arrays(p):
    """everything a trainable surface (easyvolcap/models/samplers/mirror_surface.py) needs: patch frame, extent grid,
    sampled height / slope fields, the global quadric (if fitted) and the LiDAR cell means (anchors)."""
    a = dict(c=p.c, R=p.R, g0=np.asarray(p.g0, float), cell=GRID, extent=p.grid.astype(np.uint8), Hh=p.Hh, Hu=p.Hu, Hv=p.Hv,
             model=p.info.get('model', ''))
    if 'quadric' in p.info: a.update(quadric=np.array(p.info['quadric']), fit_scale=np.array(p.info['fit_scale']))
    if getattr(p, 'cells', None) is not None:
        a.update(cells=p.cells[0], cells_rim_dist=p.cells[1], cells_w=p.cells[2], rim_drop=p.info.get('rim_drop', RIM_DROP))
    return a


def export_patch_ply(p, i, d):
    """patch surface as a mesh over its extent (vertices = grid cell centres) + the fit's cell means."""
    H2, W2 = p.grid.shape
    idx = -np.ones((H2, W2), np.int64)
    cy, cx = np.nonzero(p.grid)
    idx[cy, cx] = np.arange(len(cy))
    u, v = p.g0[0] + (cx + 0.5) * GRID, p.g0[1] + (cy + 0.5) * GRID
    V = p.c + np.stack([u, v, p.h(u, v)], 1) @ p.R.T
    a, b, c_, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = (a >= 0) & (b >= 0) & (c_ >= 0) & (e >= 0)
    F = np.concatenate([np.stack([a[q], c_[q], b[q]], 1), np.stack([b[q], c_[q], e[q]], 1)])
    Nw = p.normal_world(u, v)
    if ((F[:, 0] >= 0).all() and len(F)):
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        if (fn * Nw[F[:, 0]]).sum() < 0: F = F[:, ::-1]
    write_ply(os.path.join(d, f'patch{i}_surface.ply'), V, F, Nw)
    np.savez(os.path.join(d, f'patch{i}.npz'), **patch_arrays(p))
    if getattr(p, 'cells', None) is not None:
        P_, dist_, _ = p.cells
        t = np.clip(dist_ / 0.15, 0, 1)                                    # red at the rim -> blue >= 15 cm inside
        write_ply(os.path.join(d, f'patch{i}_cells.ply'), P_, C=np.stack([255 * (1 - t), 64 * np.ones_like(t), 255 * t], 1))


def candidate_points(files, T, train, min_zones):
    pts, fid, stats, sig = [], [], [], []
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lmn.OPEN_ZONES, lmn.OPEN_ZONES))
    for k in train:
        n, r1, p1 = lmn.nearest_echo(files[k])
        m = cv2.morphologyEx((n >= 2).astype(np.uint8), cv2.MORPH_OPEN, k3)
        nc, lab = cv2.connectedComponents(m, connectivity=4)
        kept = 0
        for c in range(1, nc):
            s = lab == c
            if s.sum() < min_zones: continue
            P = p1[s] @ T[k][:3, :3].T + T[k][:3, 3]
            cc, R = frame_pca(P, T[k][:3, 3])
            _, sg, inl = fit_height((P - cc) @ R)
            if inl.mean() < 0.7 or sg > SMOOTH_MAD: continue                # not a smooth surface: mixed pixels, clutter
            pts.append(P[inl]); fid.append(np.full(inl.sum(), k)); kept += inl.sum(); sig.append(sg)
        stats.append(((n >= 2).sum(), kept))
    return np.concatenate(pts), np.concatenate(fid), np.array(stats), float(np.median(sig))


def voxel_average(P, O, vox):
    key = np.floor(P / vox).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    S = np.zeros((len(cnt), 3)); So = np.zeros((len(cnt), 3))
    np.add.at(S, inv, P); np.add.at(So, inv, O)
    return S / cnt[:, None], So / cnt[:, None], cnt.astype(np.float64)


def cell_average(L, O, g0, clip=3.0):
    """average the patch points on the 1 cm (u, v) grid of the patch frame (2D surface cells, not 3D voxels: a 3D voxel
    smaller than the range noise keeps the full normal-direction scatter). Per cell: points whose residual to the
    cell median height is within clip * MAD are averaged (all three local coordinates). Returns local means, mean sensor
    origins and counts. With ~60 echoes per cell the normal-direction noise drops from ~2.5 cm to a few mm, which
    removes the noise-induced curvature bias of the algebraic MLS (c480: kappa 1.6-3.5 instead of 1.0 at h 0.15)."""
    iu, iv = np.floor((L[:, 0] - g0[0]) / GRID).astype(np.int64), np.floor((L[:, 1] - g0[1]) / GRID).astype(np.int64)
    key = iv * (iu.max() + 2) + iu
    o = np.argsort(key, kind='stable')
    key, Ls, Os = key[o], L[o], O[o]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    ends = np.r_[starts[1:], len(key)]
    Lm, Om, cnt = [], [], []
    for a, b in zip(starts, ends):
        z = Ls[a:b, 2]
        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med)) + 1e-4
        m = np.abs(z - med) < clip * mad
        if m.sum() == 0: continue
        Lm.append(Ls[a:b][m].mean(0)); Om.append(Os[a:b][m].mean(0)); cnt.append(m.sum())
    return np.array(Lm), np.array(Om), np.array(cnt, np.float64)


def bilinear(G, g0, u, v):
    """sample grid G (H, W) at local (u, v) (cell centres at g0 + (i + 0.5) * GRID), clamped at the border."""
    H, W = G.shape
    x = np.clip((u - g0[0]) / GRID - 0.5, 0, W - 1.001)
    y = np.clip((v - g0[1]) / GRID - 0.5, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * G[y0, x0] + fx * (1 - fy) * G[y0, x0 + 1] + (1 - fx) * fy * G[y0 + 1, x0]
            + fx * fy * G[y0 + 1, x0 + 1])


class GridPatch(lmn.Patch):
    """Mirror patch with a sampled height field Hh(u, v) and slope fields (from the MLS normals) in the frame (c, R)."""

    def __init__(self, c, R, Hh, Hu, Hv, grid, g0, info):
        super().__init__(c, R, np.zeros(6), grid, g0, info)
        self.Hh, self.Hu, self.Hv = Hh, Hu, Hv
        self.w0 = float(np.median(Hh[grid]))

    def h(self, u, v):
        return bilinear(self.Hh, self.g0, u, v)

    def dh(self, u, v):
        return bilinear(self.Hu, self.g0, u, v), bilinear(self.Hv, self.g0, u, v)

    def intersect(self, o, d, iters=8, tol=0.002):
        """as Patch.intersect, plus a convergence check: rays that miss the surface still end the Newton steps at some
        (u, v) inside the grid (the plane version is exact and never needed one) -> require |w - h(u, v)| < tol."""
        ol = (np.broadcast_to(o, d.shape) - self.c) @ self.R
        dl = d @ self.R
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (self.w0 - ol[:, 2]) / dl[:, 2]
            for _ in range(iters):
                p = ol + t[:, None] * dl
                hu, hv = self.dh(p[:, 0], p[:, 1])
                t = t - (p[:, 2] - self.h(p[:, 0], p[:, 1])) / (dl[:, 2] - hu * dl[:, 0] - hv * dl[:, 1])
            p = ol + t[:, None] * dl
            res = np.abs(p[:, 2] - self.h(p[:, 0], p[:, 1]))
        ok = np.isfinite(t) & (t > 0) & (res < tol)
        iu = np.floor((p[:, 0] - self.g0[0]) / GRID).astype(np.int64)
        iv = np.floor((p[:, 1] - self.g0[1]) / GRID).astype(np.int64)
        H, W = self.grid.shape
        ok &= (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        ok[ok] = self.grid[iv[ok], iu[ok]]
        return t, ok, p[:, 0], p[:, 1]

    def corners_world(self):
        """3D points on the outline of the extent (for the image bounding box of the patch)."""
        cy, cx = np.nonzero(self.grid)
        u, v = self.g0[0] + (cx + 0.5) * GRID, self.g0[1] + (cy + 0.5) * GRID
        k = np.unique(np.r_[np.argmin(u), np.argmax(u), np.argmin(v), np.argmax(v), np.arange(0, len(u), max(1, len(u) // 400))])
        return self.c + np.stack([u[k], v[k], self.h(u[k], v[k])], 1) @ self.R.T


def fit_patches(P, F, origins, sigma, min_pts=2000, min_frames=5, min_area=0.05, h=H_MLS, tau=TAU, mode='taubin',
                surface='mls', rim_drop=RIM_DROP):
    key = np.floor(P / VOX_CLUSTER).astype(np.int64)
    k0 = key.min(0)
    occ = np.zeros(tuple(key.max(0) - k0 + 1), bool)
    occ[tuple((key - k0).T)] = True
    lab, nl = ndimage.label(occ, np.ones((3, 3, 3)))
    pl = lab[tuple((key - k0).T)]
    sizes = np.bincount(pl, minlength=nl + 1)
    patches = []
    for c in np.argsort(-sizes):
        if c == 0 or sizes[c] < min_pts: continue
        sel = pl == c
        if len(np.unique(F[sel])) < min_frames: continue
        Q, Fq = P[sel], F[sel]
        Oq = origins[Fq]
        c0, R = frame_pca(Q, Oq.mean(0))
        L = (Q - c0) @ R
        g0 = L[:, :2].min(0) - 0.03
        W2, H2 = (np.ceil((L[:, :2].max(0) + 0.03 - g0) / GRID)).astype(int)
        # extent: half the interior point density (range noise and the beam footprint smear points past the edge)
        cnt = np.zeros((H2, W2), np.float32)
        iu, iv = ((L[:, 0] - g0[0]) / GRID).astype(int), ((L[:, 1] - g0[1]) / GRID).astype(int)
        np.add.at(cnt, (iv, iu), 1)
        den = cv2.GaussianBlur(cnt, (0, 0), 2.0)
        core = den > np.percentile(den[den > 0], 50)
        # extent = convex hull of the cells above 1/4 of the interior density (the half-density rule of the planar tool cut
        # off the side of the c480 mirror that the trajectory sees least: mask recall 74%)
        o2 = (den > 0.25 * np.median(den[core])).astype(np.uint8)
        o2 = cv2.morphologyEx(o2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        nc2, lab2 = cv2.connectedComponents(o2, connectivity=8)
        if nc2 < 2: continue
        big0 = (lab2 == 1 + np.argmax(np.bincount(lab2.ravel())[1:])).astype(np.uint8)
        cs, _ = cv2.findContours(big0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(big0); cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cs))], 1)
        big = hull.astype(bool)
        area = big.sum() * GRID * GRID
        if area < min_area: continue
        # global reference fits (info only): plane, sphere
        rms_plane = float(np.sqrt(np.mean(L[:, 2] ** 2)))
        A = np.c_[2 * Q, np.ones(len(Q))]
        s = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
        sc, sR = s[:3], float(np.sqrt(max(s[3] + s[:3] @ s[:3], 0)))
        rms_sphere = float(np.sqrt(np.mean((np.linalg.norm(Q - sc, axis=1) - sR) ** 2)))
        # MLS on 1 cm surface-cell means (weights = echo counts). Not fitted (only extrapolated by the MLS): cells within
        # RIM_DROP of the extent boundary, where the beam footprint mixes the rim with the frame (c480 raw echoes sit
        # 2-4 cm behind the sphere from ~2.5 cm inside the rim), and cells with < 1/4 of the median echo count
        La, Oa, wa = cell_average(L, Oq, g0)
        dist = ndimage.distance_transform_edt(big) * GRID
        ci = np.clip(np.floor((La[:, :2] - g0) / GRID).astype(np.int64), 0, [W2 - 1, H2 - 1])
        dcell = dist[ci[:, 1], ci[:, 0]]
        cells = (c0 + La @ R.T, dcell, wa.copy())                          # for --export-ply (world points, rim distance, echo counts)
        use = (dcell >= rim_drop) & (wa >= 0.25 * np.median(wa))
        La, Oa, wa = La[use], Oa[use], wa[use]
        if surface == 'quadric':
            Hh, Hu, Hv, bad, info_s = fit_global_quadric(La, wa, g0, H2, W2, big)
            info = dict(n_pts=int(len(Q)), n_cells=int(len(La)), n_frames=int(len(np.unique(Fq))), area_m2=float(area),
                        rms_plane_mm=1000 * rms_plane, rms_sphere_mm=1000 * rms_sphere, sphere_center=sc.tolist(), sphere_R=sR,
                        model='quadric_global', rim_drop=rim_drop, center=c0.tolist(), normal=R[:, 2].tolist(), **info_s)
            gp = GridPatch(c0, R, Hh, Hu, Hv, big, g0, info); gp.cells = cells
            patches.append(gp)
            continue
        Pa = c0 + La @ R.T
        Na = qm.pca_normals(Pa, h, Oa, K=1024, pw=wa)
        # sigma = single-echo noise: the weights are echo counts, so the tau prior sees the true amount of data
        kw = dict(K=1024, model='quadric', tau=tau, robust=2, pw=wa, sigma=sigma, mode=mode)
        Na = qm.fit(Pa, Na, Pa, h, **kw)['n']                                  # one refinement pass
        # height field on the grid (cells inside the extent dilated by 6 cm, for the Newton steps near the rim)
        ext = cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)
        cy, cx = np.nonzero(ext)
        gu, gv = g0[0] + (cx + 0.5) * GRID, g0[1] + (cy + 0.5) * GRID
        coef, _, _ = fit_height((Pa - c0) @ R)
        q = c0 + np.stack([gu, gv, poly4(gu, gv) @ coef], 1) @ R.T
        f = qm.fit(Pa, Na, q, h, **kw)
        xl, nl_ = (f['x'] - c0) @ R, f['n'] @ R
        nl_ = nl_ * np.sign(nl_[:, 2:3] + 1e-12)
        hu, hv = -nl_[:, 0] / nl_[:, 2], -nl_[:, 1] / nl_[:, 2]
        hh = xl[:, 2] + hu * (gu - xl[:, 0]) + hv * (gv - xl[:, 1])           # back to the cell centre (u, v)
        bad = ~f['ok'] | (nl_[:, 2] < 0.2)
        Hh, Hu, Hv = (np.zeros((H2, W2)) for _ in range(3))
        fill = np.zeros((H2, W2), bool)
        Hh[cy[~bad], cx[~bad]], Hu[cy[~bad], cx[~bad]], Hv[cy[~bad], cx[~bad]] = hh[~bad], hu[~bad], hv[~bad]
        fill[cy[~bad], cx[~bad]] = True
        # cells without a valid fit: nearest valid value (keeps the clamped bilinear sampling sane)
        _, (ny, nx) = ndimage.distance_transform_edt(~fill, return_indices=True)
        Hh, Hu, Hv = Hh[ny, nx], Hu[ny, nx], Hv[ny, nx]
        kin = f['k'][~bad & big[cy, cx]]
        info = dict(n_pts=int(len(Q)), n_cells=int(len(Pa)), n_frames=int(len(np.unique(Fq))), area_m2=float(area),
                    rms_plane_mm=1000 * rms_plane, rms_sphere_mm=1000 * rms_sphere, sphere_center=sc.tolist(), sphere_R=sR,
                    k1_med=float(np.median(kin[:, 0])), k2_med=float(np.median(kin[:, 1])),
                    mls_fail=int(bad.sum()), model='quadric_mls', mode=mode, h=h, tau=tau, center=c0.tolist(), normal=R[:, 2].tolist())
        gp = GridPatch(c0, R, Hh, Hu, Hv, big, g0, info); gp.cells = cells
        patches.append(gp)
    return patches


def fit_global_quadric(La, wa, g0, H2, W2, big):
    """--surface quadric: one implicit quadric over the (rim-dropped) cell means La (patch frame), sampled on the full
    grid. Coordinates are centred / scaled for the fit. Returns Hh, Hu, Hv, failed-cell count, info."""
    m = La.mean(0); s = np.abs(La - m).max()
    q, sg, ww = fit_quadric((La - m) / s, wa)
    # back to patch-frame coordinates: f(x) = q . v((x - m) / s); sample on the grid with the scaled form
    cy, cx = np.mgrid[0:H2, 0:W2]
    gu, gv = g0[0] + (cx.ravel() + 0.5) * GRID, g0[1] + (cy.ravel() + 0.5) * GRID
    coef, _, _ = fit_height(La)
    z0 = poly4(gu, gv) @ coef
    # one root branch over the extent; two sheets within 2 cm of each other everywhere = degenerate fit (flat mirror) -> mid-surface
    z, hu, hv, ok = quadric_height(q, (gu - m[0]) / s, (gv - m[1]) / s, (z0 - m[2]) / s, branch_mask=big.ravel(), par_tol=0.02 / s)
    z = z * s + m[2]                                                        # slopes are scale invariant
    Hh, Hu, Hv = np.zeros(H2 * W2), np.zeros(H2 * W2), np.zeros(H2 * W2)
    Hh[ok], Hu[ok], Hv[ok] = z[ok], hu[ok], hv[ok]
    fill = ok.reshape(H2, W2)
    _, (ny, nx) = ndimage.distance_transform_edt(~fill, return_indices=True)
    Hh, Hu, Hv = (G.reshape(H2, W2)[ny, nx] for G in (Hh, Hu, Hv))
    # curvature summary at the extent cells (principal curvatures of the height field from finite differences)
    huu = np.gradient(Hu, GRID, axis=1); hvv = np.gradient(Hv, GRID, axis=0); huv = np.gradient(Hu, GRID, axis=0)
    kmean = 0.5 * (huu + hvv)[big]
    info = dict(quadric=q.tolist(), fit_scale=[*m.tolist(), float(s)], resid_mad_mm=float(1000 * sg * s),
                inlier_frac=float((ww > 0).mean()), mean_curv_med=float(np.median(kmean)),
                grid_fail=int((~ok.reshape(H2, W2) & big).sum()))
    return Hh, Hu, Hv, int((~ok.reshape(H2, W2) & big).sum()), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root')
    ap.add_argument('blender_ds')
    ap.add_argument('--suffix', default='lidarnorm')
    ap.add_argument('--min-zones', type=int, default=30)
    ap.add_argument('--band', type=float, default=1.0, help='boundary band half-width, in LiDAR zone spacings')
    ap.add_argument('--h', type=float, default=H_MLS)
    ap.add_argument('--tau', type=float, default=TAU)
    ap.add_argument('--mode', default='taubin', choices=['taubin', 'oriented'], help='quadric_mls fit (see quadric_mls.fit_chunk)')
    ap.add_argument('--eval-mirror-normal', type=float, nargs=3, default=None, help='GT plane normal, evaluation only')
    ap.add_argument('--eval-sphere', type=float, nargs=4, default=None, help='GT convex sphere cx cy cz R, evaluation only')
    ap.add_argument('--preview', type=int, nargs='*', default=None, help='frames for debug previews (default: 8 evenly spaced)')
    ap.add_argument('--only', type=int, nargs='*', default=None, help='process only these frames (quick test)')
    ap.add_argument('--surface', default='mls', choices=['mls', 'quadric'], help='local quadric MLS (default) or one global implicit quadric')
    ap.add_argument('--rim-drop', type=float, default=RIM_DROP, help='outer ring of the extent left out of the fit (m)')
    ap.add_argument('--fit-only', action='store_true', help='fit + surface eval (+ --export-ply) only, no per-camera output')
    ap.add_argument('--export-ply', default=None, help='directory for the fitted surface mesh / cell means (MeshLab)')
    a = ap.parse_args()
    src = os.path.abspath(a.data_root.rstrip('/'))
    out = src + '_' + a.suffix
    dbg = out + '_debug'
    if not a.fit_only: os.makedirs(os.path.join(out, 'normals'), exist_ok=True); os.makedirs(dbg, exist_ok=True)
    if not a.fit_only: os.makedirs(os.path.join(out, 'envs'), exist_ok=True)
    for f in ('images', 'sparse', 'intri.yml', 'extri.yml'):
        if not a.fit_only and not os.path.lexists(os.path.join(out, f)): os.symlink(os.path.join(src, f), os.path.join(out, f))
    if not a.fit_only and os.path.exists(os.path.join(src, 'envs', 'points3D.ply')):
        shutil.copy2(os.path.join(src, 'envs', 'points3D.ply'), os.path.join(out, 'envs', 'points3D.ply'))

    names = sorted(os.listdir(os.path.join(src, 'normals')))
    fe, fi = cv2.FileStorage(os.path.join(src, 'extri.yml'), 0), cv2.FileStorage(os.path.join(src, 'intri.yml'), 0)
    Rc = {n: fe.getNode(f'Rot_{n}').mat() for n in names}
    Tc = {n: fe.getNode(f'T_{n}').mat().ravel() for n in names}
    Kc = {n: fi.getNode(f'K_{n}').mat() for n in names}
    files, T = lmn.load_lidar(a.blender_ds)
    assert len(files) == len(names), (len(files), len(names))
    origins = np.array([t[:3, 3] for t in T])
    train = [k for k in range(len(files)) if k % 8 != 0]          # EnvGS split: every 8th frame is val
    preview = a.preview if a.preview is not None else list(range(0, len(files), len(files) // 8))

    P, F, st, sigma = candidate_points(files, T, train, a.min_zones)
    print(f'candidates: {len(P)} points from {len(np.unique(F))} train frames '
          f'(multi-echo zones {st[:, 0].sum()}, kept after opening/components/smoothness {st[:, 1].sum()}); '
          f'height-field residual MAD (median over components) {100 * sigma:.2f} cm', flush=True)
    patches = fit_patches(P, F, origins, sigma, h=a.h, tau=a.tau, mode=a.mode, surface=a.surface, rim_drop=a.rim_drop)
    for i, p in enumerate(patches):
        print(f'patch {i}:', json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v)) for k, v in p.info.items()}), flush=True)
    D = lmn.zone_dirs(files)
    patches = [p for p in patches if lmn.coverage_ok(p, files, T, D, train, sigma)]
    dvox = lmn.direct_voxels(files, T, train[::3])
    keep = []
    for p in patches:
        st_ = lmn.seen_through_ratio(p, files, T, D, train, sigma, dvox)
        p.info['far_echo_on_direct_surface'] = float(st_)
        print(f"patch at {np.round(p.c, 2).tolist()}: {st_:.2f} of far echoes land on directly measured surfaces "
              f"-> {'rejected (see-through)' if st_ > lmn.SEE_THROUGH else 'mirror'}", flush=True)
        if st_ <= lmn.SEE_THROUGH: keep.append(p)
    patches = keep
    assert patches, 'no mirror patch found'
    spacing = np.radians(120 / 256)

    gt_n = sph = None
    if a.eval_mirror_normal is not None:
        gt_n = np.array(a.eval_mirror_normal, float); gt_n /= np.linalg.norm(gt_n)
    if a.eval_sphere is not None:
        sph = (np.array(a.eval_sphere[:3]), float(a.eval_sphere[3]))
    gt_normal = None
    if gt_n is not None:
        gt_normal = lambda X: np.broadcast_to(gt_n, X.shape)
    elif sph is not None:
        gt_normal = lambda X: (X - sph[0]) / np.linalg.norm(X - sph[0], axis=-1, keepdims=True)
    if gt_normal is not None:
        # the fitted surface itself vs GT, on the patch's own grid (interior / outer 5 cm ring)
        for i, p in enumerate(patches):
            cy, cx = np.nonzero(p.grid)
            u, v = p.g0[0] + (cx + 0.5) * GRID, p.g0[1] + (cy + 0.5) * GRID
            X = p.c + np.stack([u, v, p.h(u, v)], 1) @ p.R.T
            ang = np.degrees(np.arccos(np.clip(np.abs((p.normal_world(u, v) * gt_normal(X)).sum(1)), -1, 1)))
            dd = ndimage.distance_transform_edt(p.grid)[cy, cx] * GRID
            ring = dd < 0.05
            ring2 = (dd >= 0.05) & (dd < 0.10)
            pos = None
            if sph is not None:
                pos = 1000 * (np.linalg.norm(X - sph[0], axis=1) - sph[1])
            msg = (f'patch {i} surface vs GT: normal med {np.median(ang[~ring]):.3f} / p95 {np.percentile(ang[~ring], 95):.3f} deg (interior), '
                   f'med {np.median(ang[ring]):.3f} / p95 {np.percentile(ang[ring], 95):.3f} deg (outer 5 cm ring)')
            msg += f', med {np.median(ang[ring2]):.3f} / p95 {np.percentile(ang[ring2], 95):.3f} deg (5-10 cm ring)'
            if pos is not None:
                msg += f'; radial position error med {np.median(pos):+.1f} mm, |err| p95 {np.percentile(np.abs(pos), 95):.1f} mm'
            print(msg, flush=True)
            p.info['surface_vs_gt'] = dict(int_med=float(np.median(ang[~ring])), int_p95=float(np.percentile(ang[~ring], 95)),
                                           ring_med=float(np.median(ang[ring])), ring_p95=float(np.percentile(ang[ring], 95)),
                                           ring2_med=float(np.median(ang[ring2])), ring2_p95=float(np.percentile(ang[ring2], 95)))

    if a.export_ply:
        os.makedirs(a.export_ply, exist_ok=True)
        for i, p in enumerate(patches): export_patch_ply(p, i, a.export_ply)
        json.dump([p.info for p in patches], open(os.path.join(a.export_ply, 'patches.json'), 'w'), indent=1)
        print('surface meshes ->', a.export_ply, flush=True)
    if a.fit_only: return

    if a.only is not None: names = [n for n in names if int(n) in a.only]
    tf = json.load(open(os.path.join(a.blender_ds, 'persp', 'transforms.json')))
    ev = []
    for n in names:
        k = int(n)
        nm = cv2.imread(os.path.join(src, 'normals', n, '000000.png'), cv2.IMREAD_UNCHANGED)
        H, W = nm.shape[:2]; scale = 65535.0 if nm.dtype == np.uint16 else 255.0
        R, t, K = Rc[n], Tc[n], Kc[n]
        o = -R.T @ t
        px = int(np.ceil(K[0, 0] * np.tan(spacing)))
        ys, xs = np.mgrid[0:H, 0:W]
        mir = np.zeros(H * W, bool); unc = np.zeros(H * W, bool); nrm = np.zeros((H * W, 3)); tbest = np.full(H * W, np.inf)
        # only pixels inside the image bounding box of the patches (+ 3 zone spacings); camera behind them -> all
        roi = np.zeros((H, W), bool)
        for p in patches:
            xc_ = p.corners_world() @ R.T + t
            if (xc_[:, 2] <= 0.05).any():
                roi[:] = True; break
            uv_ = xc_[:, :2] / xc_[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            m_ = 3 * px
            x0, y0 = np.clip(np.floor(uv_.min(0) - m_).astype(int), 0, [W, H]); x1, y1 = np.clip(np.ceil(uv_.max(0) + m_).astype(int), 0, [W, H])
            roi[y0:y1, x0:x1] = True
        ri = np.flatnonzero(roi)
        dw = np.full((H * W, 3), np.nan)
        if len(ri):
            dc = np.stack([(xs.ravel()[ri] + 0.5 - K[0, 2]) / K[0, 0], (ys.ravel()[ri] + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(ri))], -1)
            dw[ri] = dc @ R; dw[ri] /= np.linalg.norm(dw[ri], axis=1, keepdims=True)
        _, r1, _ = lmn.nearest_echo(files[k])
        dz = D.reshape(-1, 3); okz = np.isfinite(dz[:, 0])
        dzw = dz[okz] @ T[k][:3, :3].T; os_ = T[k][:3, 3]; r1z = r1.reshape(-1)[okz]
        for p in patches:
            if not p.front(o) or not len(ri): continue
            tp, inside, u, v = np.full(H * W, np.inf), np.zeros(H * W, bool), np.zeros(H * W), np.zeros(H * W)
            tp[ri], inside[ri], u[ri], v[ri] = p.intersect(o, dw[ri])
            cand = inside & (tp < tbest)
            tbest[cand] = tp[cand]; mir[cand] = False; unc[cand] = False
            if not cand.any(): continue
            tz, iz, _, _ = p.intersect(os_, dzw)
            occ = iz & (r1z < tz - max(0.05, lmn.OCC_K * sigma))
            og = np.zeros(okz.shape, np.float32); og[np.where(okz)[0][occ]] = 1
            og = og.reshape(192, 256)
            og = og * (cv2.boxFilter(og, -1, (3, 3), normalize=False) >= 3)
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
        if gt_normal is not None:
            g = cv2.imread(os.path.join(a.blender_ds, 'persp', 'masks', f'mask_{k:04d}.png'), cv2.IMREAD_UNCHANGED)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST) == 3
            ii = inner.reshape(-1)
            X = o + tbest[ii][:, None] * dw[ii]                             # inner pixels are all inside the ROI
            gn = gt_normal(X) @ R.T
            ang = np.degrees(np.arccos(np.clip(np.abs((ncam * gn).sum(1)), -1, 1))) if inner.any() else np.zeros(0)
            ev.append(dict(frame=k, gt=int(g.sum()), inner=int(inner.sum()), tp=int((inner & g).sum()), band=int(band.sum()),
                           band_gt=int((band & g).sum()), iou=float((mir.astype(bool) & g).sum() / max(1, (mir.astype(bool) | g).sum())),
                           ang_med=float(np.median(ang)) if len(ang) else np.nan, ang_p99=float(np.percentile(ang, 99)) if len(ang) else np.nan))
            if k in preview:
                rgb = cv2.imread(os.path.join(src, 'images', n, '000000.png'))
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                ov = rgb.copy()
                ov[inner & g] = (0.5 * ov[inner & g] + [0, 127, 0]).astype(np.uint8)
                ov[inner & ~g] = (0.5 * ov[inner & ~g] + [127, 0, 0]).astype(np.uint8)
                ov[g & ~inner & ~band] = (0.5 * ov[g & ~inner & ~band] + [0, 0, 127]).astype(np.uint8)
                ov[band] = (0.5 * ov[band] + [0, 127, 127]).astype(np.uint8)
                nm8 = (nm / (scale / 255.0)).astype(np.uint8) if nm.dtype != np.uint8 else nm
                rs8 = (res / (scale / 255.0)).astype(np.uint8) if res.dtype != np.uint8 else res
                row = np.concatenate([cv2.resize(x, (648, 486), interpolation=cv2.INTER_AREA) for x in (rgb, ov, nm8, rs8)], 1)
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
