# Synthetic check of the 2DGS distortion output of diff_surfel_rasterization_wet_ch05 (the rasterizer EnvGS uses):
# 3 large fronto-parallel surfels at z = 2, 3, 4.5 in front of a camera at the origin. At the centre pixel the Gaussian
# falloff is ~1, so the weights are w_i = a_i T_i and the expected distortion is sum_{j<i} w_i w_j (m_i - m_j)^2 with
# m(z) = far/(far-near) (1 - near/z), near 0.2, far 100 (the CUDA code). Then autograd vs central finite differences
# (float64 FD is not possible in CUDA float32, so the surfels are kept far from every threshold).
import math
import torch
from diff_surfel_rasterization_wet_ch05 import GaussianRasterizationSettings, GaussianRasterizer

dev = 'cuda'; H = W = 64; fov = math.radians(40); zn, zf = 0.01, 100.
t = math.tan(fov / 2)
P = torch.zeros(4, 4); P[0, 0] = 1 / t; P[1, 1] = 1 / t; P[3, 2] = 1; P[2, 2] = zf / (zf - zn); P[2, 3] = -zf * zn / (zf - zn)
view = torch.eye(4, device=dev); proj = P.T.to(dev); full = view @ proj
S = GaussianRasterizationSettings(image_height=H, image_width=W, tanfovx=t, tanfovy=t, bg=torch.zeros(3, device=dev), scale_modifier=1.,
                                  viewmatrix=view, projmatrix=full, sh_degree=0, campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
R = GaussianRasterizer(S)
z = torch.tensor([2.0, 3.0, 4.5], device=dev)
xyz = torch.stack([torch.zeros(3, device=dev), torch.zeros(3, device=dev), z], 1).contiguous()
op = torch.tensor([[0.5], [0.4], [0.6]], device=dev)
sc = torch.full((3, 2), 3.0, device=dev)  # 3 m: falloff ~1 over the whole image
rot = torch.tensor([[1., 0, 0, 0]] * 3, device=dev)


def run(xyz, op, sc, C=None):
    m2d = torch.zeros_like(xyz, requires_grad=True)
    last = None
    for ch in (5, 3, 4, 6, 7):
        try:
            col = torch.rand(3, ch, device=dev) if C is None else C
            out = R(means3D=xyz, means2D=m2d, opacities=op, colors_precomp=col, scales=sc, rotations=rot)
            return out, col
        except Exception as e:
            last = e
    raise last


(img, radii, allmap, *_), col = run(xyz, op, sc)
print('colour channels', col.shape[1], 'allmap', tuple(allmap.shape))
dist = allmap[6]
c = (H // 2, W // 2)
near, far = 0.2, 100.
m = far / (far - near) * (1 - near / z)
a = op[:, 0]; T = torch.cumprod(torch.cat([torch.ones(1, device=dev), 1 - a[:-1]]), 0); w = a * T
exp_ = sum(w[i] * w[j] * (m[i] - m[j]) ** 2 for i in range(3) for j in range(i))
print(f'centre pixel: dist {dist[c].item():.6e}  expected (falloff 1) {exp_.item():.6e}  ratio {dist[c].item() / exp_.item():.4f}')
print(f'  depth (expected) {allmap[0][c].item():.4f} vs {float((w * z).sum()):.4f}; alpha {allmap[1][c].item():.4f} vs {float(w.sum()):.4f}; '
      f'median {allmap[5][c].item():.3f}')

# autograd vs FD of L = mean(dist) w.r.t. z, opacity (pre-activation not involved: raw opacity), scale
for name, base in (('z', xyz), ('opacity', op), ('scale', sc)):
    p = base.clone().requires_grad_(True)
    args = dict(xyz=xyz, op=op, sc=sc); args[{'z': 'xyz', 'opacity': 'op', 'scale': 'sc'}[name]] = p
    L = run(**args, C=col)[0][2][6].mean()
    g = torch.autograd.grad(L, p)[0]
    for k in range(3):
        idx = (k, 2) if name == 'z' else ((k, 0))
        e = 1e-3 if name != 'scale' else 1e-2
        vals = []
        for s in (1, -1):
            q = base.clone(); q[idx] += s * e; a2 = dict(xyz=xyz, op=op, sc=sc); a2[{'z': 'xyz', 'opacity': 'op', 'scale': 'sc'}[name]] = q
            vals.append(run(**a2, C=col)[0][2][6].mean().item())
        fd = (vals[0] - vals[1]) / (2 * e)
        print(f'  d mean(dist) / d {name}[{k}]: autograd {g[idx].item():+.5e}  FD {fd:+.5e}  ratio {fd / g[idx].item() if g[idx] != 0 else float("nan"):.4f}')
