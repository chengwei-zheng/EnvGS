"""Which loss terms drive the mirror normal? Per-term gradients of a trained EnvGS mirror_b run (2026-10-25).

Loads the final checkpoint in training mode, runs training forwards on training views that see the mirror (iteration set
to --iter, so the loss schedule / bfilt / perc are as at that iteration; no optimizer step, no densification) and
backpropagates every weighted loss term separately:

A. Rendered normal on mirror pixels (mirror mask == 255, world space): tangential gradient g_t (only that changes the
   normal, the losses normalise norm_map). |g_t| summed over mirror pixels = force; the component of -g_t towards the GT
   mirror normal = corrective part (> 0: gradient descent turns the normal towards the GT).
B. Restoring torque under a tilt of the mirror by +-delta about two in-plane axes (vertical, horizontal):
   dL/d(delta) = <dL/dq, dq/d(delta)> (+ <dL/dx, dx/d(delta)> for the rigid tilt) over the mirror surfels
   (|plane distance| < 1 cm inside the SM_Mirror rectangle). 'surfel': surfels turned in place (normal only),
   'rigid': the whole mirror turned about its centre (normals + positions). Restoring: dL/d(delta) has the sign of delta.
Weighted terms: img (L1) / ssim / perc from VolumetricVideoSupervisor, norm (prior) / gs_norm (depth-normal) from
EnvGSSupervisor, lidar_dpt from LidarDepthSupervisor; their sum is checked against the total loss.

    python mirror_normal_grads.py <run> [--iter 59000] [--views 8] [--delta 2]   (EnvGS repo root, envgs env)
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--iter', type=int, default=59000)
ap.add_argument('--views', type=int, default=8)
ap.add_argument('--delta', type=float, default=2.0, help='tilt (deg)')
ap.add_argument('--scene', default='/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300_lamppanel/lidar_scene.npz')
a = ap.parse_args()
sys.argv = ['main.py', '-t', 'train', '-c', f'configs/exps/envgs/mirror_b/envgs_{a.run}.yaml',
            f'exp_name=envgs/mirror_b/envgs_{a.run}', 'dry_run=True']

import numpy as np
import torch
import torch.nn.functional as F
import easyvolcap.scripts.main as M
from easyvolcap.utils.data_utils import add_iter, to_cuda

# GT mirror: plane + in-plane rectangle from the SM_Mirror triangles
sc = np.load(a.scene, allow_pickle=True)
names = json.loads(str(sc['object_names'].item()))
V = sc['tris'][sc['tri_obj'] == names.index('SM_Mirror')].reshape(-1, 3).astype(np.float64)
C = V.mean(0); _, _, vt = np.linalg.svd(V - C); N = vt[2]; U1 = np.array([0., 0., 1.]); U1 -= (U1 @ N) * N; U1 /= np.linalg.norm(U1); U2 = np.cross(N, U1)
lo, hi = ((V - C) @ np.stack([U1, U2], 1)).min(0), ((V - C) @ np.stack([U1, U2], 1)).max(0)
dev = 'cuda'
Ct, Nt = torch.tensor(C, dtype=torch.float, device=dev), torch.tensor(N, dtype=torch.float, device=dev)
AXES = {'vertical': torch.tensor(U1, dtype=torch.float, device=dev), 'horizontal': torch.tensor(U2, dtype=torch.float, device=dev)}

runner = M.train(M.cfg)
runner.load_model()
model = runner.model
model.train()
pcd = model.sampler.pcd
subs = list(model.supervisor.supervisors)
by = {type(s).__name__: s for s in subs}
vvs, envs, mrs, lds = by['VolumetricVideoSupervisor'], by['EnvGSSupervisor'], by['MirrorRegionSupervisor'], by.get('LidarDepthSupervisor')

with torch.no_grad():
    x = pcd._xyz
    d = (x - Ct) @ Nt
    uv = torch.stack([(x - Ct) @ AXES['vertical'], (x - Ct) @ AXES['horizontal']], 1)
    lo_t, hi_t = torch.tensor(lo, device=dev) - 0.02, torch.tensor(hi, device=dev) + 0.02
    msel = (d.abs() < 0.01) & (uv >= lo_t).all(1) & (uv <= hi_t).all(1)
print(f'mirror: centre {np.round(C, 3)}, normal {np.round(N, 4)}, size {np.round(hi - lo, 3)} m; {int(msel.sum())} mirror surfels of {len(msel)}', flush=True)


def qmul(p, q):
    w1, x1, y1, z1 = p.unbind(-1); w2, x2, y2, z2 = q.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def terms(out, batch):
    s = out.scalar_stats
    t = {'img': vvs.img_loss_weight * s.img_loss, 'ssim': vvs.ssim_loss_weight * s.ssim_loss}
    if 'perc_loss' in s: t['perc'] = vvs.perc_loss_weight * s.perc_loss
    if 'norm_loss' in s: t['norm'] = envs.norm_loss_weight * s.norm_loss
    if 'gs_norm_loss' in s: t['gs_norm'] = envs.gs_norm_loss_weight * s.gs_norm_loss
    if 'lidar_dpt_loss' in s: t['lidar_dpt'] = lds.lidar_dpt_loss_weight * s.lidar_dpt_loss
    tot = out.loss.mean()
    rest = tot - sum(t.values())
    if abs(rest.item()) > 1e-4 * max(abs(tot.item()), 1e-6): t['other'] = rest
    return t


# training views that see the mirror
views = []
for i, b in enumerate(runner.dataloader):
    m = mrs.mask(b.meta.camera_index[0].item(), b.meta.H[0].item(), b.meta.W[0].item(), dev) == 255
    if m.sum() > 20000: views.append(b)
    if len(views) >= a.views or i > 20 * a.views: break
print(f'{len(views)} training views with the mirror, iter {a.iter}', flush=True)


def forward(b):
    b = to_cuda(add_iter(b, a.iter, runner.total_iter))
    if hasattr(model, 'prepare_params'): model.prepare_params(runner, b)
    return model(b), b


# A: rendered-normal gradients on the mirror pixels at the trained state
A = {}
for b in views:
    out, b = forward(b)
    T = terms(out, b)
    m = (mrs.mask(b.meta.camera_index[0].item(), b.meta.H[0].item(), b.meta.W[0].item(), dev) == 255)[None]
    n = F.normalize(out.norm_map, dim=-1)[m]
    gt = Nt * torch.sign(n @ Nt)[:, None]
    e = gt - (gt * n).sum(-1, keepdim=True) * n
    en = e.norm(dim=-1, keepdim=True)
    err = torch.rad2deg(torch.asin(en.clamp(max=1)))
    eh = e / en.clamp(min=1e-9)
    for k, v in T.items():
        g = torch.autograd.grad(v, out.norm_map, retain_graph=True, allow_unused=True)[0]
        if g is None: g = torch.zeros_like(out.norm_map)
        g = g[m]; g = g - (g * n).sum(-1, keepdim=True) * n
        r = A.setdefault(k, dict(force=0., corr=0., value=0.))
        r['force'] += g.norm(dim=-1).sum().item(); r['corr'] += (-(g * eh).sum(-1)).sum().item(); r['value'] += v.item()
    A.setdefault('_err', []).append(err.median().item())
    del out
print('\nA. rendered normal on mirror pixels (sum over pixels, mean over views; corr > 0 = towards GT normal)')
print(f'   median normal error vs GT mirror plane: {np.mean(A.pop("_err")):.3f} deg')
print(f'   {"term":10s} {"loss":>10s} {"|g_t|":>11s} {"corr":>11s} {"corr/|g_t|":>10s}')
nv = len(views)
for k, r in A.items():
    print(f'   {k:10s} {r["value"] / nv:10.5f} {r["force"] / nv:11.4e} {r["corr"] / nv:11.4e} {r["corr"] / max(r["force"], 1e-30):10.3f}')

# B: restoring torque under a tilt of the mirror
q0, x0 = pcd._rotation.data.clone(), pcd._xyz.data.clone()
idx = msel.nonzero()[:, 0]
B = {}
for mode in ('surfel', 'rigid'):
    for ax, av in AXES.items():
        for sg in (-1, 1):
            dl = np.deg2rad(sg * a.delta)
            qa = torch.cat([torch.tensor([np.cos(dl / 2)], device=dev), np.sin(dl / 2) * av]).float()
            with torch.no_grad():
                pcd._rotation.data[idx] = qmul(qa[None].expand(len(idx), 4), q0[idx])
                if mode == 'rigid':
                    p = x0[idx] - Ct
                    c, s_ = np.cos(dl), np.sin(dl)
                    pcd._xyz.data[idx] = Ct + p * c + torch.cross(av[None].expand_as(p), p, dim=-1) * s_ + av * (p @ av)[:, None] * (1 - c)
                dq = 0.5 * qmul(torch.cat([torch.zeros(1, device=dev), av])[None].expand(len(idx), 4), F.normalize(pcd._rotation.data[idx], dim=-1))
                dq = dq * pcd._rotation.data[idx].norm(dim=-1, keepdim=True)
                dx = torch.cross(av[None].expand(len(idx), 3), pcd._xyz.data[idx] - Ct, dim=-1)
            for b in views:
                out, b = forward(b)
                for k, v in terms(out, b).items():
                    gq, gx = torch.autograd.grad(v, [pcd._rotation, pcd._xyz], retain_graph=True, allow_unused=True)
                    tq = (gq[idx] * dq).sum().item() if gq is not None else 0.
                    tx = (gx[idx] * dx).sum().item() if (gx is not None and mode == 'rigid') else 0.
                    B.setdefault((mode, ax, sg), {}).setdefault(k, 0.)
                    B[(mode, ax, sg)][k] += (tq + tx) / nv
                del out
            with torch.no_grad():
                pcd._rotation.data.copy_(q0); pcd._xyz.data.copy_(x0)

keys = sorted({k for v in B.values() for k in v}, key=lambda k: ['img', 'ssim', 'perc', 'norm', 'gs_norm', 'lidar_dpt', 'other'].index(k))
print(f'\nB. dL/d(delta) (1/rad, mean over views) at a tilt of -{a.delta}/+{a.delta} deg; restoring = sign of delta (-/+); '
      f'"rest" = (slope(+) - slope(-)) / 2, > 0 restoring')
print(f'   {"mode":7s} {"axis":10s} ' + ' '.join(f'{k:>27s}' for k in keys + ['total']))
for mode in ('surfel', 'rigid'):
    for ax in AXES:
        cells = []
        for k in keys + ['total']:
            f = lambda sg: sum(B[(mode, ax, sg)].values()) if k == 'total' else B[(mode, ax, sg)].get(k, 0.)
            cells.append(f'{f(-1):+8.4f} {f(1):+8.4f} {(f(1) - f(-1)) / 2:+8.4f}')
        print(f'   {mode:7s} {ax:10s} ' + ' '.join(f'{c:>27s}' for c in cells))
json.dump(dict(run=a.run, iter=a.iter, views=nv, A=A, B={'|'.join(map(str, k)): v for k, v in B.items()}),
          open(f'data/result/envgs/mirror_b/envgs_{a.run}/mirror_normal_grads_it{a.iter}_d{a.delta:g}.json', 'w'), indent=1)
