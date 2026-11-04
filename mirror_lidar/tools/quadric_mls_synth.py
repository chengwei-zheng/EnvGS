"""Synthetic test of quadric_mls.py (2026-10-28): LiDAR-like noisy points on known surfaces, normal / position error.

    ~/miniconda3/envs/envgs/bin/python mirror_lidar/tools/quadric_mls_synth.py --out mirror_lidar/experiments/2026-10-28_quadric_mls_synth

Surfaces (height fields z = g(u, v) in a local frame, z toward the sensors): plane disc, convex sphere cap R = 1 m (the
c480 mirror: aperture radius 0.6 m), convex cylinders r = 0.5 / 1 m (vertical axis, 1.2 m high, +-55 deg arc),
elliptic paraboloid (radii 1 / 2 m, a non-sphere quadric) and a wavy free-form (1 cm sine bumps, 0.6 m wavelength).
Sensor: 16 poses 1.5-3.5 m away, 0-60 deg off-axis, TH-S02-like angular spacing 0.47 deg per zone; points are the clean
surface points pushed along the sensor ray by Gaussian range noise sigma (synthetic TH-S02 ~2.5 cm, real ~1 cm).
Optional outliers: a fraction of the points pushed 5-30 cm behind the surface along the ray.
Input normals = weighted PCA in a ball of radius h, oriented to the sensor; optionally one refinement pass (re-fit at
every data point, use those normals). Queries = 3000 random data points; GT = closest point on the true surface.
Bands: interior = > 0.15 m from the boundary, edge = < 0.05 m.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quadric_mls as qm  # noqa: E402

ZONE = np.radians(0.47)


class Surf:
    def __init__(self, name, g, gu, gv, domain, bdist, area):
        self.name, self.g, self.gu, self.gv, self.domain, self.bdist, self.area = name, g, gu, gv, domain, bdist, area

    def X(self, uv):
        return np.stack([uv[:, 0], uv[:, 1], self.g(uv[:, 0], uv[:, 1])], 1)

    def N(self, uv):
        n = np.stack([-self.gu(uv[:, 0], uv[:, 1]), -self.gv(uv[:, 0], uv[:, 1]), np.ones(len(uv))], 1)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def sample(self, n, rng):
        out = []
        while sum(len(o) for o in out) < n:
            uv = self.domain(rng, 4 * n)
            j = np.sqrt(1 + self.gu(uv[:, 0], uv[:, 1]) ** 2 + self.gv(uv[:, 0], uv[:, 1]) ** 2)
            out.append(uv[rng.random(len(uv)) < j / j.max() * 0.999])
        return np.concatenate(out)[:n]

    def closest(self, y, uv0, iters=8):
        uv = uv0.copy()
        for _ in range(iters):
            r = self.X(uv) - y
            Ju = np.stack([np.ones(len(uv)), np.zeros(len(uv)), self.gu(uv[:, 0], uv[:, 1])], 1)
            Jv = np.stack([np.zeros(len(uv)), np.ones(len(uv)), self.gv(uv[:, 0], uv[:, 1])], 1)
            a, b, c = (Ju * Ju).sum(1), (Ju * Jv).sum(1), (Jv * Jv).sum(1)
            ru, rv = (Ju * r).sum(1), (Jv * r).sum(1)
            det = a * c - b * b
            uv = uv - np.stack([(c * ru - b * rv) / det, (a * rv - b * ru) / det], 1)
        return uv


def disc(a):
    def dom(rng, n):
        r, t = a * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
        return np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return dom, (lambda uv: a - np.linalg.norm(uv, axis=1))


def rect(hu, hv):
    def dom(rng, n):
        return np.stack([hu * (2 * rng.random(n) - 1), hv * (2 * rng.random(n) - 1)], 1)
    return dom, (lambda uv: np.minimum(hu - np.abs(uv[:, 0]), hv - np.abs(uv[:, 1])))


def surfaces():
    S = []
    dom, bd = disc(0.6)
    z = lambda u, v: 0 * u
    S.append(Surf('plane', z, z, z, dom, bd, np.pi * 0.36))
    R = 1.0
    S.append(Surf('sphere_R1', lambda u, v: np.sqrt(R * R - u * u - v * v) - R,
                  lambda u, v: -u / np.sqrt(R * R - u * u - v * v), lambda u, v: -v / np.sqrt(R * R - u * u - v * v),
                  dom, bd, 2 * np.pi * R * (R - np.sqrt(R * R - 0.36))))
    for r in (0.5, 1.0):
        hu = r * np.sin(np.radians(55))
        d2, b2 = rect(hu, 0.6)
        S.append(Surf(f'cylinder_r{r:g}', (lambda r: lambda u, v: np.sqrt(r * r - u * u) - r)(r),
                      (lambda r: lambda u, v: -u / np.sqrt(r * r - u * u))(r), lambda u, v: 0 * u,
                      d2, b2, 1.2 * r * np.radians(110)))
    R1, R2 = 1.0, 2.0
    S.append(Surf('paraboloid_1_2', lambda u, v: -(u * u / (2 * R1) + v * v / (2 * R2)),
                  lambda u, v: -u / R1, lambda u, v: -v / R2, dom, bd, np.pi * 0.36 * 1.06))
    A, L = 0.01, 0.6
    k = 2 * np.pi / L
    S.append(Surf('wavy_1cm', lambda u, v: A * np.sin(k * u) * np.sin(k * v),
                  lambda u, v: A * k * np.cos(k * u) * np.sin(k * v), lambda u, v: A * k * np.sin(k * u) * np.cos(k * v),
                  dom, bd, np.pi * 0.36))
    return S


def sensors(rng, n=16):
    d = rng.uniform(1.5, 3.5, n)
    th = np.radians(rng.uniform(0, 60, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([d * np.sin(th) * np.cos(ph), d * np.sin(th) * np.sin(ph), d * np.cos(th)], 1)


def simulate(s, sig, outl, rng):
    Ps, Ss, UV = [], [], []
    for o in sensors(rng):
        n = int(s.area * np.cos(np.arctan2(np.linalg.norm(o[:2]), o[2])) / (ZONE * np.linalg.norm(o)) ** 2)
        uv = s.sample(n, rng)
        X = s.X(uv)
        vis = ((o - X) * s.N(uv)).sum(1) > 0
        uv, X = uv[vis], X[vis]
        d = X - o
        rng_ = np.linalg.norm(d, axis=1, keepdims=True)
        e = rng.normal(0, sig, (len(X), 1))
        if outl > 0:
            m = rng.random(len(X)) < outl
            e[m, 0] += rng.uniform(0.05, 0.30, m.sum())
        Ps.append(o + d / rng_ * (rng_ + e))
        Ss.append(np.broadcast_to(o, X.shape))
        UV.append(uv)
    return np.concatenate(Ps), np.concatenate(Ss), np.concatenate(UV)


def stats(err_deg, pos, bd):
    out = {}
    for band, m in (('int', bd > 0.15), ('edge', bd < 0.05)):
        if m.sum() < 20:
            continue
        out[band] = dict(med=float(np.median(err_deg[m])), p95=float(np.percentile(err_deg[m], 95)),
                         pos_rms_mm=float(1000 * np.sqrt(np.mean(pos[m] ** 2))), n=int(m.sum()))
    return out


METHODS = [('plane', dict(model='plane')), ('sphere', dict(model='sphere')),
           ('quad_free', dict(model='quadric', tau=None)), ('quad_t3', dict(model='quadric', tau=3.0)),
           ('quad_t1', dict(model='quadric', tau=1.0)), ('quad_t0.3', dict(model='quadric', tau=0.3))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sigmas', type=float, nargs='*', default=[0.01, 0.025])
    ap.add_argument('--hs', type=float, nargs='*', default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument('--outliers', type=float, default=0.0)
    ap.add_argument('--robust', type=int, default=0)
    ap.add_argument('--refine', type=int, default=1, help='normal refinement passes (re-fit at all data points)')
    ap.add_argument('--beta', type=float, default=0.002)
    ap.add_argument('--true-sigma', type=int, default=0, help='pass the simulated sigma instead of estimating it')
    ap.add_argument('--n-query', type=int, default=3000)
    ap.add_argument('--max-pool', type=int, default=20000)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in surfaces():
        if a.only and s.name not in a.only:
            continue
        for sig in a.sigmas:
            rng = np.random.default_rng(0)
            P, O, UV = simulate(s, sig, a.outliers, rng)
            if len(P) > a.max_pool:
                k = rng.choice(len(P), a.max_pool, replace=False)
                P, O, UV = P[k], O[k], UV[k]
            qi = rng.choice(len(P), a.n_query, replace=False)
            for h in a.hs:
                t0 = time.time()
                tree = qm.cKDTree(P)
                ip = qm.neighbours(tree, P, P, h, 256, np.random.default_rng(1))
                iq = ip[qi]
                N0 = qm.pca_normals(P, h, O, idx=ip)
                for mname, kw in METHODS:
                    kw = dict(kw, beta=a.beta, robust=a.robust, sigma=sig if a.true_sigma else None)
                    N = N0
                    for _ in range(a.refine):
                        f = qm.fit(P, N, P, h, idx=ip, **kw)
                        N = np.where(f['ok'][:, None], f['n'], N0)
                    f = qm.fit(P, N, P[qi], h, idx=iq, **kw)
                    uvg = s.closest(f['x'], UV[qi])
                    Xg, Ng = s.X(uvg), s.N(uvg)
                    ang = np.degrees(np.arccos(np.clip((f['n'] * Ng).sum(1), -1, 1)))
                    pos = ((f['x'] - Xg) * Ng).sum(1)
                    ok = f['ok']
                    st = stats(ang[ok], pos[ok], s.bdist(uvg[ok]))
                    row = dict(surf=s.name, sigma=sig, h=h, method=mname, n_pts=len(P), fail=int((~ok).sum()),
                               k_med=np.median(f['k'][ok & (s.bdist(uvg) > 0.15)], 0).round(3).tolist(), **st)
                    rows.append(row)
                    i, e = st.get('int', {}), st.get('edge', {})
                    print(f"{s.name:15s} sig {100 * sig:.1f}cm h {h:.2f} {mname:10s} "
                          f"int {i.get('med', np.nan):5.2f}/{i.get('p95', np.nan):5.2f} deg {i.get('pos_rms_mm', np.nan):5.1f}mm  "
                          f"edge {e.get('med', np.nan):5.2f}/{e.get('p95', np.nan):5.2f} deg {e.get('pos_rms_mm', np.nan):5.1f}mm  "
                          f"k {row['k_med']} fail {row['fail']}", flush=True)
                print(f'  ({time.time() - t0:.0f}s)', flush=True)
    json.dump(dict(args=vars(a), rows=rows), open(os.path.join(a.out, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
