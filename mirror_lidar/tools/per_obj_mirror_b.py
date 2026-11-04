# per-object depth / PSNR + specular weight on the mirror, scene B runs (1296x972, scale 1 = GT poses)
import os, sys, glob, json, cv2, numpy as np
from eval_synthetic_glass import read_exr_depth
P = os.environ.get('GT_PERSP', '/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b/dataset_b300/persp')  # r300: GT_PERSP=<proj_indoor_mirror_b_rot30/dataset_r300/persp>
E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/mirror_b/envgs_'
tj = json.load(open(P + '/transforms.json')); on = tj['object_names']
sub = lambda a: a[1::2, 1::2][:972, :1296]
runs = sys.argv[1:]
for r in runs:
    acc = {}
    spec = []
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]; pr = np.load(f)
        g = cv2.resize(cv2.imread(P + '/' + fr['file_path'])[..., ::-1], (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        se = ((pr['rgb'].astype(np.float32) / 255 - g) ** 2).mean(-1)
        mask = sub(cv2.imread(P + '/' + fr['mask_path'], -1)); obj = sub(cv2.imread(P + '/' + fr['obj_id_path'], -1))
        dg = sub(read_exr_depth(P + '/' + fr['depth_file_path'], 1944, 2592)); dp = pr['depth']
        sp = cv2.imread(f'{E}{r}/SPECULAR/frame0000_camera{fid:04d}.png', -1)
        if sp is not None:
            sp = cv2.resize(sp, (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / (65535 if sp.dtype == np.uint16 else 255)
            if sp.ndim == 3: sp = sp[..., :3].mean(-1)
            m = mask == 3
            if m.sum() > 100: spec.append((sp[m].mean(), sp[(mask != 3) & (obj > 0)].mean()))
        for o in np.unique(obj):
            if o == 0: continue
            rr = obj == o
            if rr.sum() < 200: continue
            a = acc.setdefault(on[o - 1], {'se': [], 'ae': [], 'n': 0})
            a['se'].append(se[rr].mean()); a['ae'].append(np.abs(dp - dg)[rr]); a['n'] += rr.sum()
    print(f'## {r}' + (f'   SPECULAR weight: mirror {np.mean([s[0] for s in spec]):.3f}  other objects {np.mean([s[1] for s in spec]):.3f}' if spec else ''))
    tot = sum(a['n'] for a in acc.values())
    for k, a in sorted(acc.items(), key=lambda kv: -kv[1]['n'])[:14]:
        ae = np.concatenate(a['ae'])
        print(f"  {k:28s} px {a['n']/tot*100:5.1f}%  PSNR {np.mean([-10*np.log10(x) for x in a['se']]):6.2f}  depth MAE {ae.mean():.3f} med {np.median(ae):.3f} m  >10cm {(ae>0.1).mean():.2f}")
