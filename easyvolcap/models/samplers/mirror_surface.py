# Explicit mirror surface for the reflection normal (2026-11-02, mirror_lidar/doc/MIRROR_SURFACE.md "可学镜面曲面").
#
# EnvGS reflects every traced pixel about the rendered 2DGS normal. On a curved mirror the surfels only reach 1.4-1.8 deg
# (p90 4-5 deg) even with the LiDAR normal prior, and the convex reflection is very sensitive to it (cv300: replacing the
# reflection geometry with the LiDAR-fitted quadric at inference alone gains 1.3 dB). Here the mirror is an explicit surface,
# initialised from the LiDAR fit (mirror_lidar/tools/lidar_mirror_normals_curved.py --export-ply -> patch<i>.npz), and the
# reflected ray of a pixel uses the surface normal where the camera ray hits the surface inside the mirror extent and close to
# the rendered depth (so occluders in front of the mirror keep their own normal). Only the reflection DIRECTION changes: the
# origin is still the rendered 2DGS depth, the surfels and all their losses are untouched.
#
# Surface models (MirrorSurface subclasses, picked by `type`, all in the patch frame x = (p - c) R, R = [u v n]):
#   QuadricSurface  one implicit quadric f = q . (x^2, y^2, z^2, xy, xz, yz, x, y, z, 1) = 0 in the fit's normalised coordinates
#                   ((x - m) / s); 10 coefficients, 9 DOF (scale-free); planes / spheres exactly
#   SplineSurface   height field z = h(x, y), uniform cubic B-spline on a control grid (spacing `ctrl`, default 2 cm), least-squares
#                   initialised from the fit's sampled height field (local MLS or quadric): local shape freedom
# `learnable` False: fixed (buffers); True: nn.Parameters, trained from `learn_start_iter` through the traced reflection
# (the tracer returns dL/d ray_d), anchored to the LiDAR cell means of the mirror interior (rim strip of the fit excluded:
# the rim echoes mix with the frame) and, for the spline, regularised on the change of its second differences.
# losses(): dict of weighted scalar losses, added by MirrorRegionSupervisor (output.msurf_loss).
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from easyvolcap.utils.console_utils import *


class MirrorSurface(nn.Module):
    """Base: patch frame, extent grid (1 cm occupancy of the fit), LiDAR anchors, reflection-normal query."""

    def __init__(self, fit: str, learnable: bool = False, learn_start_iter: int = 4500, tol: float = 0.15,
                 anchor_weight: float = 0.0, anchor_sigma: float = 0.002, anchor_huber: float = 3.0,
                 eval_sphere: list = None, extent_dilate: float = 0.0, **kwargs):
        super().__init__()
        a = np.load(fit)
        self.fit = fit
        self.learnable, self.learn_start_iter, self.tol = learnable, learn_start_iter, tol
        self.anchor_weight, self.anchor_sigma, self.anchor_huber = anchor_weight, anchor_sigma, anchor_huber
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float)
        self.register_buffer('c', f32(a['c']), persistent=False)
        self.register_buffer('R', f32(a['R']), persistent=False)
        self.register_buffer('g0', f32(a['g0']), persistent=False)
        self.cell = float(a['cell'])
        ext = a['extent'].astype(bool)
        if extent_dilate > 0:  # 2026-11-06, c2_480: the LiDAR extent misses a 1-5.5 cm strip along the least-seen rim (and the top cap)
            from scipy import ndimage
            n0 = ext.sum(); ext = ndimage.binary_dilation(ext, iterations=int(round(extent_dilate / float(a['cell']))))
            log(f'extent dilated by {extent_dilate * 100:.0f} cm: {n0} -> {ext.sum()} cells')
        self.register_buffer('extent', torch.as_tensor(ext), persistent=False)
        use = (a['cells_rim_dist'] >= float(a['rim_drop'])) & (a['cells_w'] >= 0.25 * np.median(a['cells_w']))
        self.register_buffer('anchor_x', f32((a['cells'][use] - a['c']) @ a['R']), persistent=False)  # patch frame
        self.register_buffer('anchor_w', f32(a['cells_w'][use] / a['cells_w'][use].mean()), persistent=False)
        self.eval_sphere = eval_sphere  # logging only: GT sphere (cx, cy, cz, R)
        self.a = a
        self.iter = 0
        log(f'{type(self).__name__}: {fit}, learnable {learnable} (from iter {learn_start_iter}), extent {int(self.extent.sum())} cells, '
            f'{int(use.sum())} anchors (rim drop {float(a["rim_drop"]):.2f} m)')

    # -- to implement --------------------------------------------------------------------------------------------------
    def intersect_local(self, ol, dl):
        """camera rays in the patch frame -> ray parameter t (in units of dl), hit converged mask. No gradient needed."""
        raise NotImplementedError

    def normal_local(self, x):
        """outward-ish (unoriented) surface normal at local points x (on the surface), differentiable w.r.t. the parameters."""
        raise NotImplementedError

    def anchor_residual(self, x):
        """signed distance (m) of local points x to the surface (anchor loss)."""
        raise NotImplementedError

    def reg_loss(self):
        return None

    # -- shared --------------------------------------------------------------------------------------------------------
    def inside(self, xl):
        ij = torch.floor((xl[..., :2] - self.g0) / self.cell).long()
        H, W = self.extent.shape
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < W) & (ij[..., 1] >= 0) & (ij[..., 1] < H)
        ij = torch.stack([ij[..., 0].clamp(0, W - 1), ij[..., 1].clamp(0, H - 1)], -1)
        return ok & self.extent[ij[..., 1], ij[..., 0]]

    def frozen(self):
        return (not self.learnable) or (not self.training) or self.iter < self.learn_start_iter

    def reflection_normals(self, ray_o: torch.Tensor, ray_d: torch.Tensor, dpt: torch.Tensor):
        """(B, P, 3) rays, (B, P, 1) rendered depth along ray_d -> world normals facing the camera (B, P, 3), mask (B, P, 1)."""
        with torch.no_grad():
            ol = (ray_o - self.c) @ self.R
            dl = ray_d @ self.R
            t, ok = self.intersect_local(ol, dl)
            x = ol + t[..., None] * dl
            hit = ok & (t > 0) & self.inside(x)
            ok = hit & ((t[..., None] - dpt).abs()[..., 0] < self.tol)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen()):
            n = F.normalize(self.normal_local(x) @ self.R.T, dim=-1)
        n = torch.where((n * ray_d).sum(-1, keepdim=True) > 0, -n, n)
        if self.frozen(): n = n.detach()
        # 2026-11-06: surface targets for the Gaussian-to-surface consistency losses (MirrorRegionSupervisor msurf_dpt / msurf_norm):
        # hit without the depth gate (the Gaussians that are off by > tol are the ones to pull back), all detached
        self.targets = dotdict(t=t[..., None], hit=hit[..., None], n=n.detach())
        return n, ok[..., None]

    def losses(self):
        out = dotdict()
        if self.frozen(): return out
        if self.anchor_weight > 0:
            r = self.anchor_residual(self.anchor_x) / self.anchor_sigma
            k = self.anchor_huber
            hub = torch.where(r.abs() < k, 0.5 * r * r, k * (r.abs() - 0.5 * k))
            out.msurf_anchor = self.anchor_weight * (hub * self.anchor_w).mean()
        reg = self.reg_loss()
        if reg is not None: out.msurf_reg = reg
        return out

    @torch.no_grad()
    def stats(self):
        """logging: anchor RMS (mm) and, with eval_sphere, the surface normal error vs the GT sphere over the extent (deg)."""
        out = dotdict()
        out.msurf_anchor_rms_mm = 1000 * self.anchor_residual(self.anchor_x).pow(2).mean().sqrt()
        if self.eval_sphere is not None:
            H, W = self.extent.shape
            jj, ii = torch.nonzero(self.extent, as_tuple=True)
            xy = self.g0 + (torch.stack([ii, jj], -1).float() + 0.5) * self.cell
            o = torch.cat([xy, torch.full_like(xy[:, :1], 1.0)], -1)          # cast down the patch normal onto the surface
            d = torch.tensor([0., 0., -1.], device=o.device).expand_as(o)
            t, ok = self.intersect_local(o, d)
            x = o + t[:, None] * d
            n = F.normalize(self.normal_local(x) @ self.R.T, dim=-1)
            X = x @ self.R.T + self.c
            g = F.normalize(X - torch.as_tensor(self.eval_sphere[:3], device=X.device, dtype=X.dtype), dim=-1)
            ang = torch.rad2deg(torch.acos((n * g).sum(-1).abs().clamp(max=1)))[ok]
            dist = torch.as_tensor(self._rim_dist(), device=X.device)[jj, ii][ok]
            if ang.numel():
                out.msurf_gt_med = ang.median()
                if (dist < 0.05).any(): out.msurf_gt_rim5 = ang[dist < 0.05].median()
        return out

    def _rim_dist(self):
        if not hasattr(self, '_rd'):
            from scipy import ndimage
            self._rd = (ndimage.distance_transform_edt(self.extent.cpu().numpy()) * self.cell).astype(np.float32)
        return self._rd


class QuadricSurface(MirrorSurface):
    def __init__(self, fit: str, **kwargs):
        super().__init__(fit, **kwargs)
        a = self.a
        assert 'quadric' in a, f'{fit}: no global quadric (fit with --surface quadric)'
        q = torch.as_tensor(a['quadric'], dtype=torch.float)
        self.register_buffer('m', torch.as_tensor(a['fit_scale'][:3], dtype=torch.float), persistent=False)
        self.s = float(a['fit_scale'][3])
        if self.learnable: self.msurf_q = nn.Parameter(q)
        else: self.register_buffer('msurf_q', q)

    def A(self):
        q = self.msurf_q
        return torch.stack([torch.stack([q[0], q[3] / 2, q[4] / 2, q[6] / 2]), torch.stack([q[3] / 2, q[1], q[5] / 2, q[7] / 2]),
                            torch.stack([q[4] / 2, q[5] / 2, q[2], q[8] / 2]), torch.stack([q[6] / 2, q[7] / 2, q[8] / 2, q[9]])])

    def hom(self, x):
        xs = (x - self.m) / self.s
        return torch.cat([xs, torch.ones_like(xs[..., :1])], -1)

    def intersect_local(self, ol, dl):
        A = self.A()
        Xo = self.hom(ol); Xd = torch.cat([dl / self.s, torch.zeros_like(dl[..., :1])], -1)
        a = ((Xd @ A) * Xd).sum(-1); b = 2 * ((Xd @ A) * Xo).sum(-1); c = ((Xo @ A) * Xo).sum(-1)
        disc = b * b - 4 * a * c
        sq = disc.clamp_min(0).sqrt()
        lin = a.abs() < 1e-12
        r1 = torch.where(lin, -c / b, (-b - sq) / (2 * a)); r2 = torch.where(lin, -c / b, (-b + sq) / (2 * a))
        lo, hi = torch.minimum(r1, r2), torch.maximum(r1, r2)
        t = torch.full_like(a, float('inf')); okall = torch.zeros_like(a, dtype=torch.bool)
        for r in (hi, lo):                                                  # nearer valid root (in front, inside the extent) wins
            x = ol + r.nan_to_num(0, 0, 0)[..., None] * dl
            ok = ((disc >= 0) | lin) & (r > 0) & self.inside(x) & (x[..., 2].abs() < 0.3)
            t = torch.where(ok, r, t); okall = okall | ok
        return t.nan_to_num(0, 0, 0), okall

    def normal_local(self, x):
        return (self.hom(x) @ self.A())[..., :3]                            # grad f (up to the 1 / s and the factor 2)

    def anchor_residual(self, x):
        X = self.hom(x)
        f = ((X @ self.A()) * X).sum(-1)
        g = 2 * (X @ self.A())[..., :3]
        return f / g.norm(dim=-1).clamp_min(1e-9) * self.s


def _bspline(t):
    """uniform cubic B-spline basis and derivative at fractional positions t (...,) -> (..., 4), (..., 4)."""
    t2, t3 = t * t, t * t * t
    b = torch.stack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3], -1) / 6
    db = torch.stack([-(1 - t) ** 2, 3 * t2 - 4 * t, -3 * t2 + 2 * t + 1, t2], -1) / 2
    return b, db


class SplineSurface(MirrorSurface):
    def __init__(self, fit: str, ctrl: float = 0.02, margin: float = 0.06, reg_weight: float = 0.0, **kwargs):
        super().__init__(fit, **kwargs)
        a = self.a
        self.ctrl, self.reg_weight = ctrl, reg_weight
        ext = a['extent'].astype(bool)
        jj, ii = np.nonzero(ext)
        xy = a['g0'] + (np.stack([ii, jj], -1) + 0.5) * a['cell']
        lo, hi = xy.min(0) - margin, xy.max(0) + margin
        nx, ny = int(np.ceil((hi[0] - lo[0]) / ctrl)), int(np.ceil((hi[1] - lo[1]) / ctrl))
        self.register_buffer('x0', torch.as_tensor(lo, dtype=torch.float), persistent=False)
        self.nx, self.ny = nx, ny
        # least-squares control values from the fit's sampled heights on the extent dilated by the margin
        from scipy import ndimage
        dil = ndimage.binary_dilation(ext, iterations=int(round(margin / a['cell'])))
        jj, ii = np.nonzero(dil)
        xy = torch.as_tensor(a['g0'] + (np.stack([ii, jj], -1) + 0.5) * a['cell'], dtype=torch.float)
        z = torch.as_tensor(a['Hh'][jj, ii], dtype=torch.float)
        C0 = torch.zeros(ny + 3, nx + 3)
        idx, wts, _, _ = self._stencil(xy)
        Am = torch.zeros(len(xy), C0.numel())
        Am.scatter_add_(1, idx.reshape(len(xy), 16), wts.reshape(len(xy), 16))
        D = self._d2_matrix()                                                 # small curvature ridge: controls outside the data
        # double + SVD driver: the default (gelsy) returned a 6-60 mm RMS solution on this rank-deficient system
        sol = torch.linalg.lstsq(torch.cat([Am, 1e-3 * D]).double(), torch.cat([z, torch.zeros(len(D))]).double()[:, None],
                                 driver='gelsd').solution[:, 0].float()
        C = sol.reshape(ny + 3, nx + 3)
        log(f'SplineSurface: {ny + 3} x {nx + 3} control points ({ctrl * 100:.0f} cm), init RMS vs fit heights '
            f'{1000 * (Am @ sol - z).pow(2).mean().sqrt():.3f} mm')
        self.register_buffer('d2_init', self._d2(C), persistent=False)
        if self.learnable: self.msurf_h = nn.Parameter(C)
        else: self.register_buffer('msurf_h', C)

    def _stencil(self, xy):
        g = (xy - self.x0.to(xy)) / self.ctrl
        i = torch.floor(g).long()
        i = torch.stack([i[..., 0].clamp(0, self.nx - 1), i[..., 1].clamp(0, self.ny - 1)], -1)
        f = (g - i.float()).clamp(0, 1)
        bx, dbx = _bspline(f[..., 0]); by, dby = _bspline(f[..., 1])
        r = torch.arange(4, device=xy.device)
        rows = i[..., 1, None] + r                                           # (..., 4)
        cols = i[..., 0, None] + r
        idx = rows[..., :, None] * (self.nx + 3) + cols[..., None, :]        # (..., 4, 4)
        w = by[..., :, None] * bx[..., None, :]
        return idx, w, (bx, dbx), (by, dby)

    def _d2(self, C):
        return torch.cat([(C[2:, 1:-1] - 2 * C[1:-1, 1:-1] + C[:-2, 1:-1]).reshape(-1), (C[1:-1, 2:] - 2 * C[1:-1, 1:-1] + C[1:-1, :-2]).reshape(-1),
                          (C[2:, 2:] - C[2:, :-2] - C[:-2, 2:] + C[:-2, :-2]).reshape(-1) / 4])

    def _d2_matrix(self):
        n = (self.ny + 3) * (self.nx + 3)
        E = torch.eye(n).reshape(n, self.ny + 3, self.nx + 3)
        return torch.stack([self._d2(E[k]) for k in range(n)], 1)            # (rows, n)

    def height(self, xy, grad=False):
        idx, w, (bx, dbx), (by, dby) = self._stencil(xy)
        Cf = self.msurf_h.reshape(-1)
        Cn = Cf[idx]                                                         # (..., 4, 4)
        h = (w * Cn).sum((-1, -2))
        if not grad: return h
        hx = (by[..., :, None] * dbx[..., None, :] * Cn).sum((-1, -2)) / self.ctrl
        hy = (dby[..., :, None] * bx[..., None, :] * Cn).sum((-1, -2)) / self.ctrl
        return h, hx, hy

    def intersect_local(self, ol, dl, iters=10, tol=0.002):
        jj, ii = torch.nonzero(self.extent, as_tuple=True)
        w0 = self.height(self.g0 + (torch.stack([ii, jj], -1).float() + 0.5) * self.cell).median()
        t = (w0 - ol[..., 2]) / dl[..., 2]
        t = t.nan_to_num(0, 0, 0)
        for _ in range(iters):
            p = ol + t[..., None] * dl
            h, hx, hy = self.height(p[..., :2], grad=True)
            den = dl[..., 2] - hx * dl[..., 0] - hy * dl[..., 1]
            t = t - ((p[..., 2] - h) / torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)).nan_to_num(0, 0, 0)
        p = ol + t[..., None] * dl
        ok = (p[..., 2] - self.height(p[..., :2])).abs() < tol
        return t, ok

    def normal_local(self, x):
        _, hx, hy = self.height(x[..., :2], grad=True)
        return torch.stack([-hx, -hy, torch.ones_like(hx)], -1)

    def anchor_residual(self, x):
        _, hx, hy = self.height(x[..., :2], grad=True)
        return (x[..., 2] - self.height(x[..., :2])) / torch.sqrt(1 + hx * hx + hy * hy)

    def reg_loss(self):
        if self.reg_weight <= 0: return None
        dk = (self._d2(self.msurf_h) - self.d2_init) / self.ctrl ** 2          # change of curvature (1/m)
        return self.reg_weight * dk.pow(2).mean()


SURFACES = dict(quadric=QuadricSurface, spline=SplineSurface)


def build_mirror_surface(type: str = 'quadric', **kwargs) -> MirrorSurface:
    return SURFACES[type](**kwargs)
