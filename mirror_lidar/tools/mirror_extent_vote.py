"""Mirror surface extent from multi-view 2D mirror masks (2026-11-09, c2_480, step 3 of the SAM mask refinement).

The LiDAR fit (patch<i>.npz, --surface quadric) gives the mirror SURFACE; its extent (LiDAR echo occupancy) misses the rim where the
beam footprint mixes with the frame and where the LiDAR sees the mirror at grazing angles (c2_480: up to 5.5 cm on the right, ~22 cm
at the top). Here every cell of the fitted surface (1 cm grid, padded and extrapolated by --pad m past the LiDAR extent) votes in
every train camera that sees it:
  - visible: in the image, the camera in front of the surface, not occluded (the same frame's nearest LiDAR echo along the LiDAR ray
    to the cell more than --occ m in front of it);
  - vote = the 2D mirror mask at the projected pixel (masks: LiDAR interior | SAM within --fuse-px of the LiDAR mask, as
    sam_mirror_mask.py 'L+<variant>'); pixels within --edge-px of the mask boundary do not vote;
  - weight = cos(incidence) * min(1, pixel footprint of a cell / --full-px) (grazing and far views count less).
Mirror evidence that does not come from the masks (the masks include the frame: SAM segments the whole mirror object):
  - LiDAR: per cell, the zones whose nearest echo lands within --lidar-tol of the surface at that cell, and the fraction of them
    with >= 2 echoes (mirror: surface echo + ghost; a frame / wall cell: single echo). Cells with >= --veto-n zones and multi-echo
    fraction < --veto-frac are vetoed (LiDAR says solid, non-mirror surface).
  - appearance: per cell, the colour spread over the views that see it (median absolute deviation of the RGB samples): a mirror
    shows different things from different views, a frame / wall does not. Cells with spread < --photo-min are vetoed (0 = off).
extent = (weighted vote fraction > --frac, weight sum >= --min-w, not vetoed), largest component touching the LiDAR extent, holes filled.
With --sphere (GT mirror_sphere.json) and --aperture (visible cap radius), evaluation only: cell precision / recall / IoU against the
GT cap and the missing / extra rim width per 30 deg sector around the mirror axis (sector 180 = right side of the viewer image).
Writes <out>/patch0.npz (the fit with the padded grid and the new extent: msurf / normal-prior input), <out>/cells.npz (per-cell
votes, weights, LiDAR and appearance stats), <out>/extent.png and <out>/report.txt.
    python mirror_lidar/tools/mirror_extent_vote.py <patch0.npz> <data_root> <blender_ds> <lidar masks> <sam dir>/<variant> <out>
        [--sphere <json> --aperture 0.669]
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_mirror_normals as lmn          # noqa: E402
import lidar_mirror_normals_curved as lmc   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('fit'); ap.add_argument('data_root'); ap.add_argument('blender_ds'); ap.add_argument('masks'); ap.add_argument('sam'); ap.add_argument('out')
ap.add_argument('--pad', type=float, default=0.12, help='candidate cells: LiDAR extent dilated by this (m)')
ap.add_argument('--fuse-px', type=int, default=40); ap.add_argument('--edge-px', type=int, default=3)
ap.add_argument('--full-px', type=float, default=4.0, help='a cell spanning this many px gets full weight')
ap.add_argument('--occ', type=float, default=0.05)
ap.add_argument('--frac', type=float, default=0.5); ap.add_argument('--min-w', type=float, default=3.0)
ap.add_argument('--lidar-tol', type=float, default=0.04); ap.add_argument('--veto-n', type=int, default=20); ap.add_argument('--veto-frac', type=float, default=0.2)
ap.add_argument('--photo-min', type=float, default=0.0)
ap.add_argument('--sphere', default=''); ap.add_argument('--aperture', type=float, default=0.669)
ap.add_argument('--all-cams', action='store_true', help='also the val cameras (every 8th); default train only')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
log_lines = []
def log(s):
    print(s, flush=True); log_lines.append(s)

# ---- padded grid on the fitted quadric ----
f = dict(np.load(a.fit))
c, R, cell = f['c'], f['R'], float(f['cell'])
ext0 = f['extent'].astype(bool)
p = int(round(a.pad / cell)) + 3
H0, W0 = ext0.shape; H, W = H0 + 2 * p, W0 + 2 * p
g0 = f['g0'] - p * cell
ext_l = np.zeros((H, W), bool); ext_l[p:p + H0, p:p + W0] = ext0
cand = ndimage.binary_dilation(ext_l, iterations=int(round(a.pad / cell)))
cy, cx = np.mgrid[0:H, 0:W]
gu, gv = g0[0] + (cx.ravel() + 0.5) * cell, g0[1] + (cy.ravel() + 0.5) * cell
q, fs = f['quadric'], f['fit_scale']; m, s = fs[:3], fs[3]
z0 = np.pad(f['Hh'], p, mode='edge').ravel()
z, hu, hv, ok = lmc.quadric_height(q, (gu - m[0]) / s, (gv - m[1]) / s, (z0 - m[2]) / s, branch_mask=ext_l.ravel())
Hh, Hu, Hv = (np.where(ok, x, 0).reshape(H, W) for x in (z * s + m[2], hu, hv))
dz = (Hh[p:p + H0, p:p + W0] - f['Hh'])[ext0]
log(f'grid {H0}x{W0} -> {H}x{W}; candidates {cand.sum()} cells (LiDAR extent {ext0.sum()}); quadric re-sampling vs the fit Hh on the '
    f'extent: max |dh| {1000 * np.abs(dz).max():.3f} mm; candidate cells without a quadric root: {(cand & ~ok.reshape(H, W)).sum()}')
cand &= ok.reshape(H, W)
ci = np.flatnonzero(cand.ravel())
Xl = np.stack([gu[ci], gv[ci], Hh.ravel()[ci]], 1)
X = c + Xl @ R.T                                                       # world points of the candidate cells
nl = np.stack([-Hu.ravel()[ci], -Hv.ravel()[ci], np.ones(len(ci))], 1); nl /= np.linalg.norm(nl, axis=1, keepdims=True)
N = nl @ R.T                                                            # patch +z faces the sensors
Nc = len(ci)

# ---- cameras, LiDAR ----
names = sorted(os.listdir(os.path.join(a.data_root, 'images')))
fe, fi = cv2.FileStorage(os.path.join(a.data_root, 'extri.yml'), 0), cv2.FileStorage(os.path.join(a.data_root, 'intri.yml'), 0)
files, T = lmn.load_lidar(a.blender_ds)
assert len(files) == len(names)
D = lmn.zone_dirs(files)
okD = np.isfinite(D[..., 0])
dtree = cKDTree(D[okD]); dflat = np.flatnonzero(okD.ravel())
cams = [k for k in range(len(names)) if a.all_cams or k % 8 != 0]

vote_w = np.zeros(Nc); tot_w = np.zeros(Nc); n_view = np.zeros(Nc, int); n_occ = np.zeros(Nc, int)
lid_n = np.zeros(Nc, int); lid_multi = np.zeros(Nc, int)
col_sum = np.zeros((Nc, 3)); col_sq = np.zeros((Nc, 3)); col_n = np.zeros(Nc)
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.fuse_px + 1,) * 2)
cell_tree = cKDTree(Xl[:, :2])
for it, k in enumerate(cams):
    n = names[k]
    Rc, tc, K = fe.getNode(f'Rot_{n}').mat(), fe.getNode(f'T_{n}').mat().ravel(), fi.getNode(f'K_{n}').mat()
    o = -Rc.T @ tc
    # LiDAR of this frame: occlusion along the LiDAR ray to each cell, and echo statistics on the surface
    ne, r1, p1 = lmn.nearest_echo(files[k])
    Tl = T[k]; ol = Tl[:3, 3]
    dl = (X - ol) @ Tl[:3, :3]; rl = np.linalg.norm(dl, axis=1); dl /= rl[:, None]
    _, j = dtree.query(dl, workers=8)
    zi = dflat[j]
    r1z = r1.reshape(-1)[zi]
    occ = np.isfinite(r1z) & (r1z < rl - a.occ)
    # echoes on the surface: nearest echo of every zone -> cell (u, v) within lidar-tol of the surface
    v1 = np.isfinite(r1.reshape(-1))
    P1 = p1.reshape(-1, 3)[v1] @ Tl[:3, :3].T + ol
    Pl = (P1 - c) @ R
    d_, jj = cell_tree.query(Pl[:, :2], distance_upper_bound=cell * 0.75, workers=8)
    hit = np.isfinite(d_)
    hit[hit] &= np.abs(Pl[hit, 2] - Xl[jj[hit], 2]) < a.lidar_tol
    np.add.at(lid_n, jj[hit], 1); np.add.at(lid_multi, jj[hit], (ne.reshape(-1)[v1][hit] >= 2).astype(int))
    # camera projection
    xc = X @ Rc.T + tc
    front = (N * (o - X)).sum(1) > 0
    vis = (xc[:, 2] > 0.05) & front
    uv = xc[:, :2] / np.maximum(xc[:, 2:3], 1e-6) @ np.diag([K[0, 0], K[1, 1]]) + K[:2, 2]
    lm = cv2.imread(os.path.join(a.masks, n, '000000.png'), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(os.path.join(a.data_root, 'images', n, '000000.png'))
    Hi, Wi = img.shape[:2]
    xi, yi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    vis &= (xi >= 0) & (xi < Wi) & (yi >= 0) & (yi < Hi)
    if lm is None: lm = np.zeros((Hi, Wi), np.uint8)
    sp_ = os.path.join(a.sam, n, '000000.png'); sm = cv2.imread(sp_, 0) if os.path.exists(sp_) else None
    mask = lm == 255
    if sm is not None and (lm > 0).any(): mask |= (sm > 0) & (cv2.dilate((lm > 0).astype(np.uint8), ker) > 0)
    mu8 = mask.astype(np.uint8)
    dist_in = cv2.distanceTransform(mu8, cv2.DIST_L2, 3); dist_out = cv2.distanceTransform(1 - mu8, cv2.DIST_L2, 3)
    sel = np.flatnonzero(vis)
    occ_s = occ[sel]; n_occ[sel[occ_s]] += 1; sel = sel[~occ_s]
    if len(sel):
        yy, xx = yi[sel], xi[sel]
        inside = mask[yy, xx]
        edge = np.where(inside, dist_in[yy, xx], dist_out[yy, xx]) <= a.edge_px
        dvec = o - X[sel]; dist = np.linalg.norm(dvec, axis=1)
        cosi = (N[sel] * dvec).sum(1) / dist
        foot = K[0, 0] * cell * cosi / xc[sel, 2]                          # projected cell size (px), foreshortened
        w = cosi * np.minimum(1, foot / a.full_px)
        use = ~edge
        vote_w[sel[use]] += w[use] * inside[use]; tot_w[sel[use]] += w[use]; n_view[sel[use]] += 1
        col = img[yy, xx].astype(np.float64) / 255
        good = foot >= 1.0                                                  # colour samples only where a cell covers >= 1 px
        col_sum[sel[good]] += col[good]; col_sq[sel[good]] += col[good] ** 2; col_n[sel[good]] += 1
    if it % 50 == 0: print(f'{it}/{len(cams)} cameras ({time.time() - t0:.0f} s)', flush=True)

frac = vote_w / np.maximum(tot_w, 1e-9)
mf = lid_multi / np.maximum(lid_n, 1)
cm = col_sum / np.maximum(col_n, 1)[:, None]
spread = np.sqrt(np.maximum(col_sq / np.maximum(col_n, 1)[:, None] - cm ** 2, 0)).mean(1)     # colour std over the views

def grid(v, fill=0):
    G = np.full(H * W, fill, dtype=np.asarray(v).dtype); G[ci] = v
    return G.reshape(H, W)
veto_l = (lid_n >= a.veto_n) & (mf < a.veto_frac)
veto_p = (col_n >= 5) & (spread < a.photo_min) if a.photo_min > 0 else np.zeros(Nc, bool)


def cleanup(sel):
    e = grid(sel, False)
    lab, nl_ = ndimage.label(e)
    keep = np.unique(lab[e & ext_l]); keep = keep[keep > 0]
    e = np.isin(lab, keep)
    return ndimage.binary_fill_holes(e)


variants = {'lidar': ext_l, 'dil4': ndimage.binary_dilation(ext_l, iterations=4) & cand}
base = (frac > a.frac) & (tot_w >= a.min_w)
variants['vote'] = cleanup(base)
variants['vote+lidarveto'] = cleanup(base & ~veto_l)
if a.photo_min > 0:
    variants['vote+photoveto'] = cleanup(base & ~veto_p)
    variants['vote+bothveto'] = cleanup(base & ~veto_l & ~veto_p)
final = 'vote+bothveto' if a.photo_min > 0 else 'vote+lidarveto'
log(f'cameras {len(cams)}; cells with weight >= {a.min_w}: {(tot_w >= a.min_w).sum()}/{Nc}; LiDAR veto {veto_l.sum()} cells, photo veto {veto_p.sum()}')

# ---- evaluation against the GT cap (evaluation only) ----
gt = None
if a.sphere:
    S = json.load(open(a.sphere)); C, Rs, AX = np.array(S['centre']), S['R'], np.array(S['axis'])
    Xa = c + np.stack([gu, gv, Hh.ravel()], 1) @ R.T - C
    ax_ = Xa @ AX; rad = np.linalg.norm(Xa - ax_[:, None] * AX, axis=1)
    gt = ((rad <= a.aperture) & (ax_ > 0)).reshape(H, W)
    U = np.cross(AX, [0, 0, 1.]); U /= np.linalg.norm(U); Vv = np.cross(U, AX)
    ang = (np.degrees(np.arctan2(Xa @ Vv, Xa @ U)) % 360).reshape(H, W); radg = rad.reshape(H, W)
    secs = [(s0, s0 + 30) for s0 in range(0, 360, 30)]
    log(f'GT cap (radius {a.aperture} m): {gt.sum()} cells; candidates cover {(gt & cand).sum()} of them')
    log('extent            |  cells   P      R      IoU  | rim per 30-deg sector around the axis (deg 0 = viewer-image left, 90 = top, 180 = right):')
    log('                  |                             | missing (-) / extra (+) width in cm, median over the sector')
    log(f'{"":18s}|{"":29s}| ' + ' '.join(f'{s0:>5d}' for s0, _ in secs))
    for name, e in variants.items():
        tp = (e & gt).sum(); P_ = tp / max(e.sum(), 1); R_ = tp / gt.sum(); I_ = tp / (e | gt).sum()
        row = []
        for s0, s1 in secs:
            sec = (ang >= s0) & (ang < s1)
            # extent outer radius along the sector: 95th percentile of the radii of its cells
            r_e = np.percentile(radg[sec & e], 99) if (sec & e).any() else 0
            row.append(f'{(r_e - a.aperture) * 100:+5.1f}')
        log(f'{name:18s}| {e.sum():6d} {P_ * 100:6.2f} {R_ * 100:6.2f} {I_ * 100:6.2f} | ' + ' '.join(row))
    gi = gt.ravel()[ci]
    ring_in = gi & (radg.ravel()[ci] > a.aperture - 0.06)
    ring_out = ~gi & (radg.ravel()[ci] < a.aperture + 0.06)
    for nm_, sel_ in (('GT mirror, outer 6 cm', ring_in), ('outside GT, within 6 cm (frame)', ring_out)):
        ww = sel_ & (tot_w >= a.min_w)
        log(f'{nm_:34s}: vote frac median {np.median(frac[ww]):.2f} (>0.5: {(frac[ww] > 0.5).mean() * 100:.0f}%), LiDAR zones median '
            f'{np.median(lid_n[sel_]):.0f}, multi-echo frac median {np.median(mf[sel_ & (lid_n > 0)]) if (sel_ & (lid_n > 0)).any() else float("nan"):.2f}, '
            f'LiDAR-vetoed {veto_l[sel_].mean() * 100:.0f}%, colour spread median {np.median(spread[sel_ & (col_n >= 5)]):.3f}')

np.savez(os.path.join(a.out, 'cells.npz'), ci=ci, H=H, W=W, frac=frac, tot_w=tot_w, n_view=n_view, n_occ=n_occ, lid_n=lid_n,
         lid_multi=lid_multi, spread=spread, col_n=col_n, **({'gt': gt} if gt is not None else {}))
out = dict(f); out.update(g0=g0, extent=variants[final].astype(np.uint8), Hh=Hh, Hu=Hu, Hv=Hv,
                          extent_note=np.array(f'mirror_extent_vote.py ({final}) from {a.masks} + {a.sam}, fuse {a.fuse_px} px'))
np.savez(os.path.join(a.out, 'patch0.npz'), **out)
for name, e in variants.items():
    np.save(os.path.join(a.out, f'extent_{name.replace("+", "_")}.npy'), e)

# ---- picture: vote fraction (grey where no weight), outlines ----
sc = 4
img = np.full((H, W, 3), 40, np.uint8)
fr = grid(frac, np.nan); tw = grid(tot_w, 0.0)
hm = cv2.applyColorMap((np.nan_to_num(fr) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
img[tw >= a.min_w] = hm[tw >= a.min_w]
img = cv2.resize(img, (W * sc, H * sc), interpolation=cv2.INTER_NEAREST)
flip = lambda x: np.ascontiguousarray(x[::-1])                        # patch v (up) at the top
img = flip(img)
for e, col in ([(gt, (255, 0, 255))] if gt is not None else []) + [(ext_l, (0, 255, 255)), (variants[final], (255, 255, 255))]:
    cs, _ = cv2.findContours(flip(cv2.resize(e.astype(np.uint8), (W * sc, H * sc), interpolation=cv2.INTER_NEAREST)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, 2)
for i, s_ in enumerate(['vote fraction (viridis), grey = weight < min', 'magenta GT cap, yellow LiDAR extent, white ' + final]):
    cv2.putText(img, s_, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(a.out, 'extent.png'), img)
open(os.path.join(a.out, 'report.txt'), 'w').write('\n'.join(log_lines) + '\n')
log(f'done ({time.time() - t0:.0f} s) -> {a.out}')
