# per_mirror.py <dataset_dir> <pred_npz_dir> <depth_key> : per-mirror-object PSNR / depth stats at 1296x972
import os, sys, glob, json, cv2, numpy as np
import OpenEXR
from eval_synthetic_glass import read_exr_depth, umeyama_sim3, frame_id, GL_TO_CV, gt_layer_stack, assign_layers, LAYERS
from datasets.colmap import Parser
D, M, key = sys.argv[1:4]; P = D + '/persp'
tj = json.load(open(P + '/transforms.json')); on = tj['object_names']
p = Parser(D + '/th_s02_format/extracted', factor=1, normalize=False, test_every=8)
gt = np.stack([np.array(tj['frames'][frame_id(n)]['transform_matrix']) @ GL_TO_CV for n in p.image_names])
s, _, _ = umeyama_sim3(p.camtoworlds[:, :3, 3], gt[:, :3, 3])
objs = {k: on.index(n) + 1 for k, n in (('round mirror', 'mirror'), ('WallMirror', 'WallMirror')) if n in on}
acc = {k: {'se': [], 'lab': [], 'derr': []} for k in objs}
sub = lambda a: a[1::2, 1::2][:972, :1296]
for f in sorted(glob.glob(M + '/*.npz')):
    fid = int(os.path.basename(f)[:4]); fr = tj['frames'][fid]; pr = np.load(f)
    g = cv2.resize(cv2.imread(P + '/' + fr['file_path'])[..., ::-1], (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    se = ((pr['rgb'].astype(np.float32) / 255 - g) ** 2).mean(-1)
    mask = sub(cv2.imread(P + '/' + fr['mask_path'], -1)); obj = sub(cv2.imread(P + '/' + fr['obj_id_path'], -1))
    dg = sub(read_exr_depth(P + '/' + fr['depth_file_path'], 1944, 2592)); L = {k: sub(v) for k, v in np.load(P + '/' + fr['gt_layers_path']).items()}
    dp = pr[key] * s; lab = assign_layers(dp, gt_layer_stack(dg, L), 0.1)
    for k, o in objs.items():
        r = (obj == o) & (mask == 3)
        if r.sum() < 100: continue
        acc[k]['se'].append(se[r].mean()); acc[k]['lab'].append(lab[r]); acc[k]['derr'].append(np.abs(dp - dg)[r])
for k, a in acc.items():
    if not a['se']: continue
    lab = np.concatenate(a['lab']); nl = LAYERS + ['ambiguous', 'none']
    print(f"  {k:12s} frames {len(a['se'])}  PSNR {np.mean([-10*np.log10(x) for x in a['se']]):.2f}  depth MAE vs mirror plane {np.concatenate(a['derr']).mean():.2f} m  |",
          '  '.join(f"{n} {(lab == j).mean():.2f}" for j, n in enumerate(nl) if (lab == j).mean() > 0.005))
