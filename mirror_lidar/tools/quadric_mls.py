"""Local algebraic-quadric MLS with a sphere prior (2026-10-28, curved-mirror normal prior, "scheme 2").

Free-form surface that reduces exactly to planes / spheres (APSS-like) and, when the data clearly asks for it, to any quadric
(cylinder, cone, paraboloid ...). For every query point q, in local coordinates x = (p - q) / h around q, fit
    f(x) = c + b.x + x^T A x                                  (10 coefficients, A symmetric)
to the oriented neighbours (p_i, n_i) by weighted linear least squares (Guennebaud & Gross 2007, oriented sphere fit):
    [ sum_i w_i f(x_i)^2  +  8 sigma^2 / (h^4 tau^2) * U(A)  +  mu * W * V(A) ] / sum_i w_i |grad f(x_i)|^2     (Taubin; default)
  or  sum_i w_i [ f(x_i)^2 + beta * |grad f(x_i) - n_i|^2 ]  +  (same priors)                    (mode='oriented', APSS)
in a local frame whose z axis is the mean input normal. The normal term fixes the scale of f (grad f ~ unit normal,
f ~ signed distance in units of h); beta is small (0.002) so the positions decide the shape and the (biased, e.g. one-sided
PCA at the boundary) input normals only fix scale and sign. U = (a_xx - a_yy)^2 / 2 + 2 a_xy^2 = (k1 - k2)^2 h^2 / 8 is
the in-surface anisotropy; with the weights w_i <= 1 acting as point counts and residuals measured in sigma, the first term
is a Gaussian prior k1 - k2 ~ N(0, tau^2): it does NOT grow with the number of points, so enough data (or low noise)
overrides it. tau = inf is a free quadric, tau -> 0 an umbilic (sphere / plane) fit. sigma: LiDAR range noise (m),
estimated from sphere-fit residuals (MAD) when not given.
V = (a_zz - (a_xx + a_yy) / 2)^2 + 2 a_xz^2 + 2 a_yz^2 ties the normal-direction coefficients, which the near-surface
data hardly constrains, to the sphere completion (keeps the system well posed; it barely changes the surface).
Exact spheres and planes have U = V = 0 and zero residual, so they are reproduced exactly whatever h is; a cylinder of
radius r is accepted once its misfit as a sphere outweighs (1 / r)^2 / tau^2.
model='plane' fixes A = 0 (oriented-plane MLS), model='sphere' fixes A = a I (APSS), model='quadric' uses tau / mu.
The query is then projected onto f = 0 (Newton along grad f) and the normal / principal curvatures are read there.
"""
import numpy as np
from scipy.spatial import cKDTree

# coefficient order: c, b_x, b_y, b_z, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz   (x^T A x = a_xx x^2 + ... + 2 a_xy x y + ...)
NC = 10


def phi(x):
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    o = np.ones_like(X)
    return np.stack([o, X, Y, Z, X * X, Y * Y, Z * Z, 2 * X * Y, 2 * X * Z, 2 * Y * Z], -1)


def grad_phi(x):
    """(..., 3, 10): row k = d phi / d x_k."""
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    o, z = np.ones_like(X), np.zeros_like(X)
    gx = np.stack([z, o, z, z, 2 * X, z, z, 2 * Y, 2 * Z, z], -1)
    gy = np.stack([z, z, o, z, z, 2 * Y, z, 2 * X, z, 2 * Z], -1)
    gz = np.stack([z, z, z, o, z, z, 2 * Z, z, 2 * X, 2 * Y], -1)
    return np.stack([gx, gy, gz], -2)


def _dev_form():
    """10x10 quadratic form u^T D u = |A - tr(A)/3 I|_F^2."""
    D = np.zeros((NC, NC))
    D[4:7, 4:7] = np.eye(3) - 1.0 / 3.0
    D[7:, 7:] = 2 * np.eye(3)
    return D


DEV = _dev_form()


def _form(*terms):
    F = np.zeros((NC, NC))
    for c, v in terms:
        v = np.asarray(v, float)
        F += c * np.outer(v, v)
    return F


_e = np.eye(NC)
UMB = _form((0.5, _e[4] - _e[5]), (2, _e[7]))
NRM = _form((1, _e[6] - 0.5 * _e[4] - 0.5 * _e[5]), (2, _e[8]), (2, _e[9]))

# the normal equations only need weighted moments sum_k w x^a y^b z^c (a+b+c <= 4) and sum_k w n_d x^e (|e| <= 1)
TERMS = [(1, (0, 0, 0)), (1, (1, 0, 0)), (1, (0, 1, 0)), (1, (0, 0, 1)), (1, (2, 0, 0)), (1, (0, 2, 0)), (1, (0, 0, 2)),
         (2, (1, 1, 0)), (2, (1, 0, 1)), (2, (0, 1, 1))]
EXPS = [(a, b, c) for a in range(5) for b in range(5) for c in range(5) if a + b + c <= 4]
EXP1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _assembly():
    E = {e: i for i, e in enumerate(EXPS)}
    E1 = {e: i for i, e in enumerate(EXP1)}
    add = lambda a, b: tuple(i + j for i, j in zip(a, b))

    def grads(t):
        c, e = t
        out = []
        for d in range(3):
            if e[d] > 0:
                ee = list(e); ee[d] -= 1
                out.append((d, c * e[d], tuple(ee)))
        return out
    Cp, Cg, Rg = np.zeros((NC * NC, len(EXPS))), np.zeros((NC * NC, len(EXPS))), np.zeros((NC, 3 * len(EXP1)))
    for i, ti in enumerate(TERMS):
        for d, c, e in grads(ti):
            Rg[i, d * len(EXP1) + E1[e]] += c
        for j, tj in enumerate(TERMS):
            Cp[i * NC + j, E[add(ti[1], tj[1])]] += ti[0] * tj[0]
            for di, ci, ei in grads(ti):
                for dj, cj, ej in grads(tj):
                    if di == dj:
                        Cg[i * NC + j, E[add(ei, ej)]] += ci * cj
    return Cp, Cg, Rg


C_PHI, C_GRAD, R_GRAD = _assembly()


def moments(x, w, n):
    P3 = []
    for d in range(3):
        p = [np.ones_like(x[..., d])]
        for _ in range(4):
            p.append(p[-1] * x[..., d])
        P3.append(p)
    mom = np.stack([(w * P3[0][a] * P3[1][b] * P3[2][c]).sum(-1) for a, b, c in EXPS], -1)
    nm = np.stack([(w * n[..., d] * P3[0][a] * P3[1][b] * P3[2][c]).sum(-1) for d in range(3) for a, b, c in EXP1], -1)
    return mom, nm
# reduced parametrisations u = T v
T_PLANE = np.eye(NC)[:, :4]
T_SPHERE = np.concatenate([np.eye(NC)[:, :4], np.array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], float).T], 1)


def coeff_A(u):
    A = np.empty(u.shape[:-1] + (3, 3))
    A[..., 0, 0], A[..., 1, 1], A[..., 2, 2] = u[..., 4], u[..., 5], u[..., 6]
    A[..., 0, 1] = A[..., 1, 0] = u[..., 7]
    A[..., 0, 2] = A[..., 2, 0] = u[..., 8]
    A[..., 1, 2] = A[..., 2, 1] = u[..., 9]
    return A


def eval_f(u, x):
    return (phi(x) * u).sum(-1)


def eval_grad(u, x):
    return u[..., :1 + 3][..., 1:] + 2 * np.einsum('...ij,...j->...i', coeff_A(u), x)


def weights(d2, h2):
    """compact quartic kernel (1 - d^2/h^2)^4 used by APSS."""
    return np.clip(1 - d2 / h2, 0, None) ** 4


def neighbours(tree, P, Q, h, K, rng):
    """K neighbours within h, padded with -1; if more than K are inside the ball a uniform random subset is taken
    (k-nearest would shrink the effective radius)."""
    lists = tree.query_ball_point(Q, h, workers=-1)
    idx = np.full((len(Q), K), -1, np.int64)
    for i, l in enumerate(lists):
        if len(l) > K:
            l = rng.choice(l, K, replace=False)
        idx[i, :len(l)] = l
    return idx


def pca_normals(P, h, sensor, K=256, seed=0, idx=None, pw=None, chunk=5000):
    """weighted PCA normals in a ball of radius h, oriented toward each point's sensor origin. pw: point weights
    (e.g. counts of voxel-averaged points)."""
    if idx is None:
        idx = neighbours(cKDTree(P), P, P, h, K, np.random.default_rng(seed))
    out = np.empty((len(P), 3))
    for i in range(0, len(P), chunk):
        ix = idx[i:i + chunk]
        ok = ix >= 0
        X = P[np.where(ok, ix, 0)] - P[i:i + chunk, None]
        w = weights((X ** 2).sum(-1), h * h) * ok
        if pw is not None:
            w = w * pw[np.where(ok, ix, 0)]
        W = w.sum(1, keepdims=True)
        mu = (w[..., None] * X).sum(1) / np.maximum(W, 1e-12)
        Xc = X - mu[:, None]
        C = np.einsum('nk,nki,nkj->nij', w, Xc, Xc)
        _, V = np.linalg.eigh(C)
        n = V[..., 0]
        s = np.sign(((sensor[i:i + chunk] - P[i:i + chunk]) * n).sum(-1))
        out[i:i + chunk] = n * np.where(s == 0, 1, s)[:, None]
    return out


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, 0.5 * c[-1])]


def estimate_sigma(P, N, h, n=2000, K=256, seed=0, pw=None):
    """robust range-noise estimate: (weighted) MAD of the residuals of local free-quadric fits at n random points (a
    sphere fit would count the shape misfit of e.g. a cylinder as noise and over-tighten the tau prior)."""
    rng = np.random.default_rng(seed)
    qi = rng.choice(len(P), min(n, len(P)), replace=False)
    idx = neighbours(cKDTree(P), P, P[qi], h, K, rng)
    f = fit_chunk(P, N, P[qi], h, idx, model='quadric', tau=None, with_res=True, pw=pw)
    ok = idx >= 0
    r = np.abs(f['res'][ok])
    return float(1.4826 * (np.median(r) if pw is None else wmedian(r, pw[idx[ok]])))


def fit(P, N, Q, h, chunk=3000, K=256, seed=0, idx=None, **kw):
    """fit_chunk over chunks of queries (memory ~ chunk * K * 40 doubles). idx: precomputed neighbours(...) of Q
    (reuse it when fitting the same queries several times)."""
    if idx is None:
        idx = neighbours(cKDTree(P), P, Q, h, K, np.random.default_rng(seed))
    if kw.get('model', 'quadric') == 'quadric' and kw.get('sigma') is None:
        kw['sigma'] = estimate_sigma(P, N, h, K=K, seed=seed, pw=kw.get('pw'))
    outs = [fit_chunk(P, N, Q[i:i + chunk], h, idx[i:i + chunk], **kw) for i in range(0, len(Q), chunk)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def fit_chunk(P, N, Q, h, idx, model='quadric', tau=1.0, mu=1.0, beta=0.002, refit=2, robust=0, sigma=None, min_nb=12,
              with_res=False, pw=None, mode='taubin'):
    """Fit at each query q in Q. Returns dict with
        x: projected points (n,3), n: unit normals (oriented like the input normals), k: principal curvatures (n,2)
        (k1 >= k2, sign: positive = surface bends away from its normal, i.e. convex toward the viewer for a convex
        mirror), r: weighted RMS residual |f| * h (m) over the neighbours, nb: neighbour count, ok: fit valid.
    refit: APSS-style iterations (refit at the projected point with the same neighbour set).
    robust: IRLS iterations with Cauchy weights on the residual distance, scale 2.5 * sigma (sigma estimated from the
    residuals if not given). pw: per-point weights (counts of voxel-averaged points); they also act as point counts in
    the tau prior.
    mode: 'taubin' (default) minimises sum w f^2 / sum w |grad f|^2 (generalized eigenproblem; the input normals only
    orient the result, so biased input normals cannot pull the shape); 'oriented' is the APSS-style linear fit with the
    normal term beta. Both overestimate the curvature when the range noise is comparable to the sag inside the kernel
    (synthetic sphere R 1 m, sigma 2.5 cm, h 0.2: kappa 1.13 oriented / 1.22 taubin; c480 raw echoes, h 0.15: 1.6-3.6);
    Taubin does NOT remove this bias (on raw noisy points it is slightly worse than 'oriented', synthetic sweep
    experiments/2026-10-28_quadric_mls_synth/taubin). What removes it is averaging the echoes first on ~1 cm surface
    cells (lidar_mirror_normals_curved.cell_average: noise 2.5 cm -> ~4 mm); on cell means both modes are equal
    (c480, h 0.2, tau 0.3: interior normal error median 0.240 deg oriented / 0.243 taubin)."""
    ok = idx >= 0
    nb = ok.sum(1)
    Pi, Ni = P[np.where(ok, idx, 0)], N[np.where(ok, idx, 0)]
    # local frame: z = weighted mean input normal; work in local metric coordinates around the query (q = 0)
    w0 = weights(((Pi - Q[:, None]) ** 2).sum(-1), h * h) * ok
    nz = (w0[..., None] * Ni).sum(1)
    nz = nz / np.maximum(np.linalg.norm(nz, axis=1, keepdims=True), 1e-12)
    a = np.where(np.abs(nz[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(nz, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    Rf = np.stack([t1, np.cross(nz, t1), nz], 2)                       # columns: t1, t2, n
    Pi = np.einsum('nkd,nde->nke', Pi - Q[:, None], Rf)
    Ni = np.einsum('nkd,nde->nke', Ni, Rf)
    Q0 = Q
    Q = np.zeros_like(Q0)
    T = {'plane': T_PLANE, 'sphere': T_SPHERE, 'quadric': np.eye(NC)}[model]
    q = Q.copy()
    rw = np.ones(idx.shape) if pw is None else pw[np.where(ok, idx, 0)] * ok
    for it in range(refit + 1 + robust):
        x = (Pi - q[:, None]) / h
        w = weights((x ** 2).sum(-1), 1.0) * ok * rw
        W = w.sum(1)
        mom, nmom = moments(x, w, Ni)
        if mode == 'taubin':
            M = T.T @ (mom @ C_PHI.T).reshape(-1, NC, NC) @ T
            Ng = T.T @ (mom @ C_GRAD.T).reshape(-1, NC, NC) @ T
        else:
            M = T.T @ (mom @ (C_PHI + beta * C_GRAD).T).reshape(-1, NC, NC) @ T
            rhs = beta * nmom @ R_GRAD.T @ T
        if model == 'quadric':
            M = M + mu * W[:, None, None] * NRM
            if tau is not None and np.isfinite(tau):
                M = M + 8 * sigma ** 2 / (h ** 4 * tau ** 2) * UMB
        m = T.shape[1]
        if mode == 'taubin':
            # min sum w f^2 / sum w |grad f|^2: eliminate c (no gradient), generalized eigenproblem on the rest
            m00 = M[:, 0, 0] + 1e-12
            Mr = M[:, 1:, 1:] - M[:, 1:, :1] @ M[:, :1, 1:] / m00[:, None, None]
            Nr = Ng[:, 1:, 1:] + (1e-9 * W[:, None, None] + 1e-12) * np.eye(m - 1)
            bad_ = ~(np.isfinite(Mr).all((1, 2)) & np.isfinite(Nr).all((1, 2))) | (nb < min_nb)
            Mr[bad_], Nr[bad_] = np.eye(m - 1), np.eye(m - 1)                  # degenerate neighbourhoods: marked not ok below
            Li = np.linalg.inv(np.linalg.cholesky(Nr))
            _, V = np.linalg.eigh(Li @ Mr @ Li.transpose(0, 2, 1))
            vr = (Li.transpose(0, 2, 1) @ V[:, :, :1])[:, :, 0]
            v = np.concatenate([-(M[:, 0, 1:] * vr).sum(1, keepdims=True) / m00[:, None], vr], 1)
            v = v * np.sqrt(W / np.maximum(np.einsum('ni,nij,nj->n', v, Ng, v), 1e-30))[:, None]   # mean |grad f|^2 = 1
            v = v * np.where((nmom @ R_GRAD.T @ T * v).sum(1) < 0, -1.0, 1.0)[:, None]            # orient like the input normals
        else:
            M = M + (1e-9 * W[:, None, None] + 1e-12) * np.eye(m)
            v = np.linalg.solve(M, rhs[..., None])[..., 0]
        u = v @ T.T
        # project the current centre (x = 0) onto f = 0
        y = np.zeros((len(Q), 3))
        for _ in range(6):
            g = eval_grad(u, y)
            y = y - (eval_f(u, y) / np.maximum((g ** 2).sum(-1), 1e-12))[:, None] * g
        if it < refit:
            q = q + h * y
            continue
        res = eval_f(u[:, None], x) / np.maximum(np.linalg.norm(eval_grad(u[:, None], x), axis=-1), 1e-6) * h   # ~ distance (m)
        if robust:
            s = sigma if sigma is not None else 1.4826 * np.median(np.abs(res[ok]))
            rw = 1.0 / (1.0 + (res / (2.5 * s)) ** 2) * (1.0 if pw is None else pw[np.where(ok, idx, 0)])
        q = q + h * y
    # read normal and curvatures at the projected point (local coords of the last fit: y)
    g = eval_grad(u, y)
    gn = np.linalg.norm(g, axis=-1, keepdims=True)
    n = g / np.maximum(gn, 1e-12)
    ref = (w[..., None] * Ni).sum(1)
    n = n * np.where((n * ref).sum(-1) < 0, -1.0, 1.0)[:, None]
    # shape operator: tangential part of Hessian (2A / h in metric units) / |grad f|
    H = 2 * coeff_A(u) / h / np.maximum(gn[..., None], 1e-12)
    b1 = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])))
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    B = np.stack([b1, np.cross(n, b1)], 2)                            # tangent basis (n,3,2)
    sgn = np.where((g * n).sum(-1) > 0, 1.0, -1.0)[:, None, None]    # f increases along n -> convex means k > 0
    k = np.linalg.eigvalsh(sgn * (B.transpose(0, 2, 1) @ H @ B))[:, ::-1]
    r = np.sqrt((w * res ** 2).sum(1) / np.maximum(w.sum(1), 1e-12))
    good = (nb >= min_nb) & np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) < h)
    out = dict(x=Q0 + np.einsum('nd,ned->ne', q, Rf), n=np.einsum('nd,ned->ne', n, Rf), k=k, r=r, nb=nb, ok=good)
    if with_res:
        out['res'] = res
    return out
