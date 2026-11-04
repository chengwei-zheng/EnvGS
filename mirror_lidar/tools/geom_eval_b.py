"""Geometry metrics of EnvGS mirror_b runs against the GT mesh (2026-10-24).

1. Fused depth point cloud: the rendered z-depth of the val views (eval_npz, 1296x972 = GT persp [1::2, 1::2]) back-projected
   with the GT cameras, vs the GT mesh / the GT depth back-projected at the same pixels (every GT-hit pixel; the mirror
   counts as its physical surface, as the first-hit depth in eval_glass_anyname).
   acc = pred point -> GT mesh, comp = GT point -> nearest pred point (union of all views), F-score at 1 / 2 / 5 cm.
2. Depth normals: normals from the cross product of the back-projected neighbours, pred vs GT (same operator), at pixels
   without a GT depth discontinuity.
3. Surfels (latest.npz): centre -> GT mesh distance, mirror-virtual fraction (> 5 cm off the mesh, < 3 cm after mirroring
   at the GT mirror plane), surfel normal vs normal of the nearest GT triangle (unsigned), for opacity > 0.5.
GT mesh: lidar_scene.npz triangles, sampled densely (point-to-plane distance of the nearest sample's triangle).

    python geom_eval_b.py <run> [<run> ...]      (run = exp tag without envgs_, e.g. r300lp_lidarnorm_shared_bfilt)
"""
import glob
import json
import os
import sys

import cv2  # before OpenEXR (gsplat env)
import numpy as np
import OpenEXR
import Imath
from scipy.spatial import cKDTree

E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data'
DS = os.environ.get('GT_DS', '/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300_lamppanel')
MC, MN = np.array([1.298, 0.249, 1.3]), np.array([0.8660254, 0.5, 0.])  # GT mirror centre / normal (r300)
# 2026-10-29, corridor c480: RESULT_GROUP=corridor, CEILING=CorridorCeiling, MIRROR_SPHERE='cx,cy,cz,R' (convex mirror: the
# 'virtual' surfel test reflects radially at the sphere, q = c + (2R - |x - c|) u, instead of at the plane MC / MN)
GROUP = os.environ.get('RESULT_GROUP', 'mirror_b')
CEILING = os.environ.get('CEILING', 'SM_Ceiling_2')
SPH = [float(v) for v in os.environ['MIRROR_SPHERE'].split(',')] if os.environ.get('MIRROR_SPHERE') else None
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
N_SAMPLES, STRIDE_P, STRIDE_G = 20_000_000, 2, 4
rng = np.random.default_rng(0)


def gt_mesh():
    d = np.load(DS + '/lidar_scene.npz', allow_pickle=True)
    T, obj, names = d['tris'].astype(np.float64), d['tri_obj'], json.loads(str(d['object_names'].item()))
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); a = np.linalg.norm(n, axis=1); n /= np.maximum(a, 1e-12)[:, None]
    f = rng.choice(len(T), N_SAMPLES, p=a / a.sum())
    u, v = rng.random(N_SAMPLES), rng.random(N_SAMPLES); s = u + v > 1; u[s], v[s] = 1 - u[s], 1 - v[s]
    S = T[f, 0] + u[:, None] * (T[f, 1] - T[f, 0]) + v[:, None] * (T[f, 2] - T[f, 0])
    print(f'GT mesh: {len(T)} triangles, {a.sum() / 2:.1f} m^2, {N_SAMPLES / 1e6:.0f}M samples', flush=True)
    return cKDTree(S), S, n[f], obj[f], names


def mesh_dist(tree, S, N, P):
    e, j = tree.query(P, workers=-1)
    dp = np.abs(((P - S[j]) * N[j]).sum(1))
    lat = np.sqrt(np.maximum(e ** 2 - dp ** 2, 0))
    return np.where(lat < 0.01, dp, e), j  # point-to-plane inside a triangle, euclidean near edges


def read_depth(p, h, w):
    return np.frombuffer(OpenEXR.InputFile(p).channel('depth.V', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)


def backproject(z, c2w, tj):
    H, W = z.shape
    ys, xs = np.mgrid[0:H, 0:W]
    x = (2 * xs + 1 + 0.5 - tj['cx']) / tj['fl_x']; y = (2 * ys + 1 + 0.5 - tj['cy']) / tj['fl_y']
    X = np.stack([x * z, y * z, z], -1)
    return X @ c2w[:3, :3].T + c2w[:3, 3]


def normals(X):
    dx, dy = X[1:-1, 2:] - X[1:-1, :-2], X[2:, 1:-1] - X[:-2, 1:-1]
    n = np.cross(dx, dy); n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    out = np.full_like(X, np.nan); out[1:-1, 1:-1] = n
    return out


def pct(x, t): return 100 * (x < t).mean()


def main():
    tree, S, SN, SO, names = gt_mesh()
    tj = json.load(open(DS + '/persp/transforms.json'))
    sub = lambda a: a[1::2, 1::2][:972, :1296]
    ceil = names.index(CEILING)
    rows = []
    for r in sys.argv[1:]:
        R = f'{E}/result/envgs/{GROUP}/envgs_{r}'
        Pp, Gp, Pm, Gm, ang, angm, Pc = [], [], [], [], [], [], []
        for f in sorted(glob.glob(R + '/eval_npz/*.npz')):
            fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
            c2w = np.array(fr['transform_matrix']) @ GL_TO_CV
            zp = np.load(f)['depth'].astype(np.float64)
            zg = sub(read_depth(DS + '/persp/' + fr['depth_file_path'], tj['h'], tj['w'])).astype(np.float64)
            mask = sub(cv2.imread(DS + '/persp/' + fr['mask_path'], -1))
            ok = np.isfinite(zg) & (zg > 0) & (mask > 0)
            okp = ok & np.isfinite(zp) & (zp > 0)
            Xp, Xg = backproject(np.where(okp, zp, 0), c2w, tj), backproject(np.where(ok, zg, 0), c2w, tj)
            st = np.zeros_like(ok); st[::STRIDE_P, ::STRIDE_P] = True
            sg = np.zeros_like(ok); sg[::STRIDE_G, ::STRIDE_G] = True
            Pp.append(Xp[okp & st]); Gp.append(Xg[ok & sg]); Pm.append(mask[okp & st] == 3); Gm.append(mask[ok & sg] == 3)
            # depth normals, skip GT discontinuities (> 2% depth jump to a 4-neighbour)
            Np, Ng = normals(Xp), normals(Xg)
            zz = np.where(ok, zg, np.nan); jump = np.zeros_like(ok)
            for a_, b_ in ((np.s_[1:-1, 2:], np.s_[1:-1, :-2]), (np.s_[2:, 1:-1], np.s_[:-2, 1:-1])):
                jump[1:-1, 1:-1] |= ~(np.abs(zz[a_] - zz[b_]) < 0.02 * zz[1:-1, 1:-1])
            okn = okp & ~jump & np.isfinite(Np[..., 0]) & np.isfinite(Ng[..., 0])
            okn[1:-1, 1:-1] &= okp[1:-1, 2:] & okp[1:-1, :-2] & okp[2:, 1:-1] & okp[:-2, 1:-1]
            c = np.clip(np.abs((Np[okn] * Ng[okn]).sum(-1)), 0, 1)
            ang.append(np.degrees(np.arccos(c))); angm.append(mask[okn] == 3)
        P, G, pm, gm = np.concatenate(Pp), np.concatenate(Gp), np.concatenate(Pm), np.concatenate(Gm)
        ang, angm = np.concatenate(ang), np.concatenate(angm)
        acc, j = mesh_dist(tree, S, SN, P)
        comp, _ = cKDTree(P).query(G, workers=-1)
        pc = SO[j] == ceil
        row = dict(run=r, n_views=len(Pp))
        for k, sel_p, sel_g in (('all', slice(None), slice(None)), ('opaque', ~pm, ~gm), ('mirror', pm, gm)):
            a, cmp_ = acc[sel_p], comp[sel_g]
            row[k] = dict(acc_med=100 * np.median(a), acc_mean=100 * a.mean(), comp_med=100 * np.median(cmp_), comp_mean=100 * cmp_.mean(),
                          chamfer=50 * (a.mean() + cmp_.mean()),
                          **{f'F@{t}': (lambda p_, r_: 2 * p_ * r_ / max(p_ + r_, 1e-9))(pct(a, t / 100), pct(cmp_, t / 100)) for t in (1, 2, 5)})
        row['ceiling_acc_med'] = 100 * np.median(acc[pc & ~pm])
        row['normal'] = dict(opaque_med=np.median(ang[~angm]), opaque_mean=ang[~angm].mean(), opaque_lt10=pct(ang[~angm], 10),
                             mirror_med=np.median(ang[angm]), mirror_lt2=pct(ang[angm], 2))
        # surfels
        g = np.load(f'{E}/trained_model/envgs/{GROUP}/envgs_{r}/latest.npz')
        xyz, op = g['sampler.pcd._xyz'], 1 / (1 + np.exp(-g['sampler.pcd._opacity'][:, 0]))
        q = g['sampler.pcd._rotation']; q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
        nrm = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)  # 3rd column of R(q)
        keep = op > 0.5
        d, j = mesh_dist(tree, S, SN, xyz[keep])
        if SPH is None:
            mq = xyz[keep] - 2 * ((xyz[keep] - MC) @ MN)[:, None] * MN
        else:
            sc, sr = np.array(SPH[:3]), SPH[3]
            rr = np.linalg.norm(xyz[keep] - sc, axis=1, keepdims=True)
            mq = sc + (2 * sr - rr) * (xyz[keep] - sc) / np.maximum(rr, 1e-9)
        dq, _ = tree.query(mq, workers=-1)
        near = d < 0.02
        sa = np.degrees(np.arccos(np.clip(np.abs((nrm[keep][near] * SN[j[near]]).sum(1)), 0, 1)))
        row['surfel'] = dict(n=len(xyz), n_op05=int(keep.sum()), med=100 * np.median(d), lt1=pct(d, .01), lt2=pct(d, .02),
                             gt5=100 * (d > .05).mean(), virtual=100 * ((d > .05) & (dq < .03)).mean(),
                             nrm_med=np.median(sa), nrm_lt10=pct(sa, 10))
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(row, open(R + '/geom_eval.json', 'w'), indent=1)

    print(f'\n{"run":46s} | fused depth points vs GT (cm, %): {"acc":>5s} {"comp":>5s} {"CD":>5s} {"F@1":>5s} {"F@2":>5s} {"F@5":>5s} '
          f'| opaque {"acc":>5s} {"F@1":>5s} {"F@2":>5s} | ceil acc | mirror acc {"F@1":>5s} '
          f'| dnormal opq med {"<10°":>5s} mir med | surfels(op>.5) {"N":>7s} {"med":>5s} {"<1cm":>5s} {">5cm":>5s} {"virt":>5s} {"nrm med":>7s} {"<10°":>5s}')
    for w in rows:
        a, o, m, n, s = w['all'], w['opaque'], w['mirror'], w['normal'], w['surfel']
        print(f'{w["run"]:46s} | {"":34s}{a["acc_med"]:5.2f} {a["comp_med"]:5.2f} {a["chamfer"]:5.2f} {a["F@1"]:5.1f} {a["F@2"]:5.1f} {a["F@5"]:5.1f} '
              f'| {"":6s} {o["acc_med"]:5.2f} {o["F@1"]:5.1f} {o["F@2"]:5.1f} | {w["ceiling_acc_med"]:8.2f} | {m["acc_med"]:10.2f} {m["F@1"]:5.1f} '
              f'| {n["opaque_med"]:15.2f} {n["opaque_lt10"]:5.1f} {n["mirror_med"]:7.2f} | {"":14s} {s["n_op05"]:7d} {s["med"]:5.2f} {s["lt1"]:5.1f} '
              f'{s["gt5"]:5.1f} {s["virtual"]:5.2f} {s["nrm_med"]:7.2f} {s["nrm_lt10"]:5.1f}')


if __name__ == '__main__':
    main()
