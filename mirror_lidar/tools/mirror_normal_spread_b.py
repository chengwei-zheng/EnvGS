# mirror normal spread, scene B runs: rendered NORMAL (test views, written at end of training) on GT mirror pixels.
# The mirror is one plane, so within a frame every mirror pixel should share one normal: report the angle to the
# per-frame median normal (convention-free: works whether NORMAL is stored in camera or world space).
# run from ~/SMBC/gsplat/gsplat/script (gsplat env):  python mirror_normal_spread_b.py <run tag> ...
import os, sys, glob, json, cv2, numpy as np
P = os.environ.get('GT_PERSP', '/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b/dataset_b300/persp')  # r300: GT_PERSP=<proj_indoor_mirror_b_rot30/dataset_r300/persp>
E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/mirror_b/envgs_'
tj = json.load(open(P + '/transforms.json'))
print(f"{'run':22s} {'frames':>6s} {'median':>7s} {'p90':>7s} {'>2deg':>6s} {'>5deg':>6s}  (deg)")
for r in sys.argv[1:]:
    ang, med = [], []
    for f in sorted(glob.glob(E + r + '/NORMAL/frame0000_camera????.png')):
        fid = int(os.path.basename(f)[-8:-4]); fr = tj['frames'][fid]
        n = cv2.imread(f, -1)
        n = n[..., :3][..., ::-1].astype(np.float64) / (65535 if n.dtype == np.uint16 else 255) * 2 - 1
        m = cv2.imread(P + '/' + fr['mask_path'], -1) == 3
        m = cv2.resize(m.astype(np.uint8), n.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
        if m.sum() < 500: continue
        v = n[m] / np.maximum(np.linalg.norm(n[m], axis=-1, keepdims=True), 1e-8)
        c = np.median(v, 0); c /= np.linalg.norm(c)
        a = np.degrees(np.arccos(np.clip(v @ c, -1, 1)))
        ang.append(a); med.append(np.median(a))
    a = np.concatenate(ang)
    print(f'{r:22s} {len(med):6d} {np.median(med):7.2f} {np.percentile(a, 90):7.2f} {(a>2).mean():6.2f} {(a>5).mean():6.2f}')
