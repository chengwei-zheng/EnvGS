# mirror flatness, scene B runs: signed depth error (rendered - GT mirror plane) on GT mirror pixels (persp/masks == 3)
#   offset = median signed error (>0: rendered surface behind the mirror), spread = per-frame std / IQR, median over frames
# run from ~/SMBC/gsplat/gsplat/script with PYTHONPATH=$PWD (gsplat env):  python mirror_flatness_b.py <run tag> ...
import os, sys, glob, json, cv2, numpy as np
from eval_synthetic_glass import read_exr_depth
P = os.environ.get('GT_PERSP', '/home/yiinqiang/SMBC/blender/project/proj_indoor_mirror_b/dataset_b300/persp')  # r300: GT_PERSP=<proj_indoor_mirror_b_rot30/dataset_r300/persp>
E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/mirror_b/envgs_'
tj = json.load(open(P + '/transforms.json'))
sub = lambda a: a[1::2, 1::2][:972, :1296]
print(f"{'run':22s} {'frames':>6s} {'offset':>8s} {'std':>7s} {'IQR':>7s} {'|e|<1cm':>8s} {'|e|<3cm':>8s}  (m)")
for r in sys.argv[1:]:
    off, std, iqr, all_e = [], [], [], []
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]
        m = sub(cv2.imread(P + '/' + fr['mask_path'], -1)) == 3
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # drop mirror-frame edge pixels
        if m.sum() < 500: continue
        e = (np.load(f)['depth'] - sub(read_exr_depth(P + '/' + fr['depth_file_path'], 1944, 2592)))[m]
        q1, q3 = np.percentile(e, [25, 75])
        off.append(np.median(e)); std.append(e.std()); iqr.append(q3 - q1); all_e.append(e)
    ae = np.abs(np.concatenate(all_e))
    print(f'{r:22s} {len(off):6d} {np.median(off):+8.3f} {np.median(std):7.3f} {np.median(iqr):7.3f} {(ae<0.01).mean():8.2f} {(ae<0.03).mean():8.2f}')
