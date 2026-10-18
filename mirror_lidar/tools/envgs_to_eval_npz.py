"""Convert EnvGS raw exports (tools/export_raw.py) into the per-frame npz layout read by
gsplat/script/eval_synthetic_glass.py --pred_dir.

EasyVolcap camera index k is the k-th image of the COLMAP model sorted by name (colmap_to_easyvolcap.py),
which is also the order eval_synthetic_glass.py uses, so camera k <-> sorted image name k.

Usage:
    python envgs_to_eval_npz.py --export_dir <data/result/<exp>/rawdepth> --images_dir <th_s02_format/extracted/images> --out_dir <dir>
Writes <out_dir>/<image_stem>.npz with rgb (H, W, 3) uint8 and depth (H, W) float32 z-depth (COLMAP units).
"""

import argparse
import glob
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_dir', required=True)
    ap.add_argument('--images_dir', required=True, help='image folder the COLMAP model was built from')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.images_dir) if n.lower().endswith(('.png', '.jpg', '.jpeg')))
    os.makedirs(args.out_dir, exist_ok=True)
    n_out = 0
    for f in sorted(glob.glob(os.path.join(args.export_dir, 'RENDER', '*.npy'))):
        cam = int(re.search(r'camera(\d+)', os.path.basename(f)).group(1))
        rgb = np.load(f)[..., :3]
        dpt = np.load(f.replace(os.sep + 'RENDER' + os.sep, os.sep + 'DEPTH' + os.sep))[..., 0]
        np.savez_compressed(os.path.join(args.out_dir, os.path.splitext(names[cam])[0] + '.npz'),
                            rgb=(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), depth=dpt.astype(np.float32))
        n_out += 1
    print(f'wrote {n_out} frames to {args.out_dir}')


if __name__ == '__main__':
    main()
