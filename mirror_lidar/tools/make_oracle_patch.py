"""ORACLE mirror patch for msurf (2026-11-08, c2_480; user: 3D surface extent from GT, with GT normals + GT masks + 2 cm START_OFFSET):
a copy of a LiDAR fit patch<i>.npz (same patch frame c / R, cell size and LiDAR anchors) whose height samples Hh and extent come
from the GT sphere cap instead of the fit:
  - grid padded by --pad m on every side (the GT cap is wider than the LiDAR extent's grid);
  - Hh / Hu / Hv = the GT sphere along the patch z axis (the spline is initialised from Hh on the extent dilated by its margin);
  - extent = cells whose sphere point lies inside the visible cap, radial distance from the mirror axis <= --aperture
    (c2_480: mirror cap 0.675 m, the frame's inner edge at 0.669 m covers the outer 6 mm).
    python mirror_lidar/tools/make_oracle_patch.py <fit patch0.npz> <mirror_sphere.json> <out patch0.npz> [--aperture 0.669] [--pad 0.1]
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('fit'); ap.add_argument('sphere'); ap.add_argument('out')
ap.add_argument('--aperture', type=float, default=0.669, help='visible cap radius around the mirror axis (m)')
ap.add_argument('--pad', type=float, default=0.1, help='grid padding (m)')
a = ap.parse_args()

f = dict(np.load(a.fit))
S = json.load(open(a.sphere))
C, Rs, AX = np.array(S['centre']), float(S['R']), np.array(S['axis'])
c, R, cell = f['c'], f['R'], float(f['cell'])
p = int(round(a.pad / cell))
H0, W0 = f['extent'].shape
H, W = H0 + 2 * p, W0 + 2 * p
g0 = f['g0'] - p * cell
jj, ii = np.mgrid[0:H, 0:W]
uv = g0 + (np.stack([ii, jj], -1) + 0.5) * cell                       # cell centres (patch frame)

# line x = c + R (u, v, h): solve |x - C|^2 = Rs^2 for h, root on the mirror side (the cap faces +axis, the patch z ~ +-axis)
q = c + uv @ R[:, :2].T - C                                            # point at h = 0, relative to the centre
ez = R[:, 2]
b = q @ ez; disc = b * b - ((q * q).sum(-1) - Rs * Rs)
ok = disc > 0
hs = np.stack([-b + np.sqrt(np.maximum(disc, 0)), -b - np.sqrt(np.maximum(disc, 0))])
on_cap = lambda h: ((q + h[..., None] * ez) @ AX) > 0                  # front half of the sphere (the cap side)
h = np.where(on_cap(hs[0]), hs[0], hs[1])
X = q + h[..., None] * ez                                               # relative to the centre
ax = X @ AX
rad = np.linalg.norm(X - ax[..., None] * AX, axis=-1)
ext = ok & (ax > 0) & (rad <= a.aperture)
# height-field slopes dh/du, dh/dv from the sphere normal n = X / Rs in the patch frame
nl = (X / Rs) @ R
Hu, Hv = -nl[..., 0] / nl[..., 2], -nl[..., 1] / nl[..., 2]
fill = lambda v: np.where(ok, v, 0.0)
print(f'grid {H0}x{W0} -> {H}x{W} (pad {p} cells); GT extent {ext.sum()} cells (LiDAR {int(f["extent"].sum())}); '
      f'extent touches the grid border: {bool(ext[0].any() or ext[-1].any() or ext[:, 0].any() or ext[:, -1].any())}; '
      f'cells without a sphere hit: {(~ok).sum()}')
old = np.zeros((H, W), bool); old[p:p + H0, p:p + W0] = f['extent'].astype(bool)
print(f'LiDAR extent cells outside the GT extent: {(old & ~ext).sum()}, GT cells outside the LiDAR extent: {(ext & ~old).sum()}')
dh = (h - np.pad(f['Hh'], p, constant_values=np.nan))[old]
print(f'GT sphere vs fit heights on the LiDAR extent: median |dh| {1000 * np.nanmedian(np.abs(dh)):.2f} mm, max {1000 * np.nanmax(np.abs(dh)):.2f} mm')
f.update(g0=g0, extent=ext.astype(np.uint8), Hh=fill(h), Hu=fill(Hu), Hv=fill(Hv), model=np.array('oracle_gt_sphere'),
         oracle_note=np.array(f'ORACLE: Hh / extent from the GT sphere {a.sphere}, visible cap radius {a.aperture} m; '
                              f'frame / anchors from {a.fit}'))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, **f)
print('saved', a.out)
