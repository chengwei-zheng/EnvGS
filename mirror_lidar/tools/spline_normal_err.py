# learned B-spline mirror normal vs the GT sphere, per extent cell, binned by horizontal position u on the mirror (same u as
# mirror_lr_bins.py: along cross(axis, z); u < 0 = right side of a camera facing the mirror) and by rim distance (2026-11-06, c2_480;
# user: the right side of the mirror in the viewer is still blurry). Rows: init (fit) and each run's checkpoint msurf_h. Writes a map png.
#     python spline_normal_err.py <fit patch0.npz> <sphere json> <out dir> <run> [<run> ...]
import json, os, sys
sys.argv, args = sys.argv[:1], sys.argv[1:]
import numpy as np, torch, torch.nn.functional as F, cv2
from easyvolcap.models.samplers.mirror_surface import SplineSurface
fit, sj, out, runs = args[0], args[1], args[2], args[3:]
S = json.load(open(sj)); C0, AX = np.array(S['centre']), np.array(S['axis'])
U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U)
s = SplineSurface(fit, learnable=False)
E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/trained_model/envgs/corridor/envgs_'
rows = [('init (fit)', s.msurf_h.clone())] + [(r, torch.as_tensor(np.load(E + r + '/latest.npz')['sampler.msurf.msurf_h'])) for r in runs]
jj, ii = torch.nonzero(s.extent, as_tuple=True)
xy = s.g0 + (torch.stack([ii, jj], -1).float() + 0.5) * s.cell
rd = torch.as_tensor(s._rim_dist())[jj, ii].numpy()
UB = [(-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6)]; RB = [(0, 2), (2, 5), (5, 10), (10, 100)]
lines = [f'{"":28s} | median normal error (deg) by u (m): ' + ' '.join(f'{a:+.1f}..{b:+.1f}' for a, b in UB) + ' | by rim dist (cm): ' + ' '.join(f'{a}-{b}' for a, b in RB) + ' | all med / p90']
H, W = s.extent.shape; tiles = []
for name, h in rows:
    s.msurf_h = h
    with torch.no_grad():
        o = torch.cat([xy, torch.full_like(xy[:, :1], 1.0)], -1); d = torch.tensor([0., 0., -1.]).expand_as(o)
        t, ok = s.intersect_local(o, d); x = o + t[:, None] * d
        n = F.normalize(s.normal_local(x) @ s.R.T, dim=-1); X = x @ s.R.T + s.c
    X, n, ok = X.numpy(), n.numpy(), ok.numpy()
    g = X - C0; g /= np.linalg.norm(g, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.abs((n * g).sum(1)), 0, 1)))
    u = (X - C0) @ U
    med = lambda m: f'{np.median(ang[m & ok]):5.2f}' if (m & ok).any() else '  -  '
    lines.append(f'{name:28s} | ' + ' '.join(f'{med((u >= a) & (u < b)):>10s}' for a, b in UB) + ' | ' + ' '.join(f'{med((rd >= a / 100) & (rd < b / 100)):>5s}' for a, b in RB)
                 + f' | {np.median(ang[ok]):.2f} / {np.percentile(ang[ok], 90):.2f}')
    m = np.zeros((H, W), np.float32); m[jj.numpy(), ii.numpy()] = np.clip(ang / 5, 0, 1) * 255
    img = cv2.applyColorMap(m.astype(np.uint8), cv2.COLORMAP_TURBO); img[~s.extent.numpy()] = 40
    img = cv2.resize(img, (W * 4, H * 4), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, name[:40], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); tiles.append(img)
cv2.imwrite(os.path.join(out, 'normal_err_0to5deg.png'), np.concatenate(tiles, 1))
# which image direction is +u in the patch map: print it
print('patch x axis . u =', float(s.R[:, 0].numpy() @ U), ' patch y axis . u =', float(s.R[:, 1].numpy() @ U))
open(os.path.join(out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
