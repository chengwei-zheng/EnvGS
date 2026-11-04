"""Which reflected rays of the convex mirror hit something within the tracer's START_OFFSET (2026-11-03, scene B_convex / cv300).

diff-surfel-tracing ignores hits closer than START_OFFSET = 0.06 m (optix_tracer/config.h) along a reflected ray, so a mirror frame
right next to the rim can be skipped. For the val cameras (every 8th frame) and a pixel grid with stride STRIDE: camera ray -> GT
sphere cap (must be the first hit of the scene, i.e. visible), reflect about the exact sphere normal, cast the reflected ray into the
scene (one BVH over all evaluated meshes) and record the first hit (distance, object) and the first hit beyond START_OFFSET
(what the tracer sees instead). Binned by distance from the rim (aperture - radial distance from the mirror axis).
Read-only (the .blend is not saved).
    blender -b proj_indoor_mirror_b_convex/indoor_mirror_b_convex.blend -P frame_hit_stats.py -- <dataset_cv300> <mirror_sphere.json> <out_dir> [mirror object]
sphere json: centre, R (+ axis, aperture; if missing they are computed from the mirror object's vertices). Mirror object default SM_Mirror
(c480: ConvexMirror, 2026-11-03).
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--') + 1:]
ds, sph_json, out = argv[:3]
MIRROR = argv[3] if len(argv) > 3 else 'SM_Mirror'
STRIDE, OFF = 6, 0.06
os.makedirs(out, exist_ok=True)
S = json.load(open(sph_json))
C, R = np.array(S['centre']), S['R']
if 'axis' not in S:   # from the mirror vertices on the sphere (|dist - R| < 2 mm)
    o = bpy.data.objects[MIRROR]; M = np.array(o.matrix_world)
    v = np.array([vv.co for vv in o.data.vertices]) @ M[:3, :3].T + M[:3, 3]
    v = v[np.abs(np.linalg.norm(v - C, axis=1) - R) < 0.002]
    ax = (v - C).mean(0); ax /= np.linalg.norm(ax)
    S['axis'] = list(ax); S['aperture'] = float(np.linalg.norm((v - C) - ((v - C) @ ax)[:, None] * ax, axis=1).max())
    print('axis / aperture from', MIRROR, np.round(ax, 4), round(S['aperture'], 4), len(v), 'sphere vertices')
AX, AP = np.array(S['axis']), S['aperture']

dg = bpy.context.evaluated_depsgraph_get()
verts, tris, owner, names = [], [], [], []
for o in bpy.context.scene.objects:
    if o.type != 'MESH' or o.hide_render: continue
    me = o.evaluated_get(dg).to_mesh()
    me.calc_loop_triangles()
    M = np.array(o.matrix_world)
    v = np.array([vv.co for vv in me.vertices]).reshape(-1, 3)
    v = v @ M[:3, :3].T + M[:3, 3]
    base = sum(len(x) for x in verts)
    t = np.array([lt.vertices[:] for lt in me.loop_triangles], dtype=np.int64).reshape(-1, 3) + base
    verts.append(v); tris.append(t); owner += [len(names)] * len(t); names.append(o.name)
    o.evaluated_get(dg).to_mesh_clear()
V = np.concatenate(verts); T = np.concatenate(tris); owner = np.array(owner)
bvh = BVHTree.FromPolygons([Vector(x) for x in V], T.tolist())
print('BVH', len(V), 'verts', len(T), 'tris', len(names), 'objects')
mirror_id = names.index(MIRROR)

tj = json.load(open(os.path.join(ds, 'persp', 'transforms.json')))
W, H, fx, fy, cx, cy = tj['w'], tj['h'], tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy']
jj, ii = np.mgrid[STRIDE // 2:H:STRIDE, STRIDE // 2:W:STRIDE]
dc = np.stack([(ii - cx) / fx, -(jj - cy) / fy, -np.ones_like(ii, float)], -1).reshape(-1, 3)   # blender camera: -z forward, y up


def cast(o, d):
    loc, n, idx, dist = bvh.ray_cast(Vector(o), Vector(d))
    return (dist, owner[idx]) if idx is not None else (np.inf, -1)


rec = []   # rim_dist, d_first, obj_first, d_after, obj_after, frame
for fi, f in enumerate(tj['frames']):
    if fi % 8: continue
    M = np.array(f['transform_matrix'])
    o = M[:3, 3]; d = dc @ M[:3, :3].T; d /= np.linalg.norm(d, axis=1, keepdims=True)
    oc = o - C; b = d @ oc; c = oc @ oc - R * R; disc = b * b - c
    ok = disc > 0
    t = -b - np.sqrt(np.where(ok, disc, 0))
    X = o + t[:, None] * d
    nrm = (X - C) / R
    rad = np.linalg.norm((X - C) - ((X - C) @ AX)[:, None] * AX, axis=1)
    ok &= (t > 0) & ((nrm @ AX) > 0) & (rad <= AP)
    for k in np.nonzero(ok)[0]:
        dist, ob = cast(o, d[k])
        if ob != mirror_id or abs(dist - t[k]) > 1e-3: continue      # sphere point hidden (frame / stand in front)
        r = d[k] - 2 * (d[k] @ nrm[k]) * nrm[k]
        p = X[k] + 1e-5 * nrm[k]
        d1, o1 = cast(p, r)
        d2, o2 = (d1, o1) if d1 >= OFF else cast(p + OFF * r, r)
        if d1 < OFF: d2 += OFF
        rec.append((AP - rad[k], d1, o1, d2, o2, fi))
    print('frame', fi, 'mirror samples', len(rec), flush=True)

A = np.array(rec, dtype=float)
np.save(os.path.join(out, 'hits.npy'), A)
json.dump(names, open(os.path.join(out, 'objects.json'), 'w')); json.dump(S, open(os.path.join(out, 'sphere.json'), 'w'), indent=1)
rd, d1, o1, d2, o2 = A[:, 0], A[:, 1], A[:, 2].astype(int), A[:, 3], A[:, 4].astype(int)
near = d1 < OFF
lines = [f'{len(A)} visible mirror samples ({len(set(A[:, 5]))} val frames, stride {STRIDE}); reflected first hit < {OFF} m: {100 * near.mean():.2f}%']
lines.append('rim-dist bin (cm) | samples | first hit < 6 cm | of those: objects (first hit) -> objects seen instead (first hit beyond 6 cm)')
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 60)]:
    m = (rd >= lo / 100) & (rd < hi / 100)
    if not m.any(): continue
    mn = m & near
    top = lambda ids: ', '.join(f'{names[i] if i >= 0 else "miss"} {100 * c / max(mn.sum(), 1):.0f}%' for i, c in
                                sorted(zip(*np.unique(ids, return_counts=True)), key=lambda x: -x[1])[:3])
    lines.append(f'{lo:>3}-{hi:<3} | {m.sum():7d} | {100 * mn.sum() / m.sum():6.2f}% | {top(o1[mn]) if mn.any() else "-"} -> {top(o2[mn]) if mn.any() else "-"}'
                 + (f' | median d first {100 * np.median(d1[mn]):.1f} cm, instead {100 * np.median(d2[mn]):.1f} cm' if mn.any() else ''))
for th in (0.01, 0.02, 0.04):
    lines.append(f'first hit < {th * 100:.0f} cm: {100 * (d1 < th).mean():.2f}%')
open(os.path.join(out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
