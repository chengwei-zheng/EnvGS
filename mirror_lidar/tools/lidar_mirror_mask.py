"""Per-camera mirror masks from a LiDAR normal-prior dataset (2026-10-21, mirror_lidar/doc/MIRROR_SURFACE.md).

lidar_mirror_normals.py writes the fitted mirror normal inside the mirror, black (0,0,0) on the eroded boundary band
and leaves the StableNormal prior everywhere else. So the mirror region is recovered exactly by comparing with the
StableNormal-only normals of the base dataset, no extra LiDAR pass needed.
Output <lidarnorm_root>/mirror_masks/<cam>/000000.png (uint8, full image resolution):
    255 = mirror interior (fitted normal), 128 = boundary band (normal loss skipped), 0 = everything else.
Not named masks/: EasyVolcap would read that as the foreground mask.

    python lidar_mirror_mask.py data/datasets/mirror_b/r300_lidarnorm_open3 data/datasets/mirror_b/r300 \
        [--eval-gt ~/SMBC/blender/project/proj_indoor_mirror_b_rot30/dataset_r300/persp]
"""
import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lidarnorm_root')
    ap.add_argument('base_root', help='dataset with the plain StableNormal normals')
    ap.add_argument('--eval-gt', default='', help='blender persp dir; compares with GT persp/masks (mirror = 3, evaluation only)')
    a = ap.parse_args()
    out = os.path.join(a.lidarnorm_root, 'mirror_masks')
    names = sorted(os.listdir(os.path.join(a.lidarnorm_root, 'normals')))
    stats = []
    for n in names:
        f = os.path.join(n, '000000.png')
        ln = cv2.imread(os.path.join(a.lidarnorm_root, 'normals', f), cv2.IMREAD_UNCHANGED)
        sn = cv2.imread(os.path.join(a.base_root, 'normals', f), cv2.IMREAD_UNCHANGED)
        band = (ln == 0).all(-1)
        mirror = (ln != sn).any(-1) & ~band
        # a fitted normal can coincide with the StableNormal value at a few pixels: close those pinholes
        mirror = cv2.morphologyEx(mirror.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & ~band
        m = np.zeros(band.shape, np.uint8)
        m[band] = 128
        m[mirror] = 255
        os.makedirs(os.path.join(out, n), exist_ok=True)
        cv2.imwrite(os.path.join(out, f), m)
        stats.append((n, mirror.mean(), band.mean()))
    nz = [s for s in stats if s[1] > 0]
    print(f'{len(names)} cameras -> {out}: {len(nz)} with mirror, mean mirror {100 * np.mean([s[1] for s in nz]):.1f}% '
          f'band {100 * np.mean([s[2] for s in nz]):.1f}% of the image (frames with mirror)')
    if a.eval_gt:
        tp = fp = fn = 0
        iou = []
        for n, _, _ in stats:
            m = cv2.imread(os.path.join(out, n, '000000.png'), cv2.IMREAD_UNCHANGED)
            g = cv2.imread(os.path.join(a.eval_gt, 'masks', f'mask_{int(n):04d}.png'), cv2.IMREAD_UNCHANGED) == 3
            p, pb = m == 255, m > 0
            tp += (p & g).sum(); fp += (p & ~g).sum(); fn += (g & ~p).sum()
            if g.any(): iou.append((pb & g).sum() / (pb | g).sum())
        r = dict(precision=float(tp / (tp + fp)), recall=float(tp / (tp + fn)), iou_with_band_median=float(np.median(iou)))
        print(f'vs GT (pixels over all frames): interior precision {100 * r["precision"]:.2f}%  recall {100 * r["recall"]:.1f}%  '
              f'IoU (interior + band) median {r["iou_with_band_median"]:.3f}')
        json.dump(r, open(os.path.join(a.lidarnorm_root, 'mirror_masks_eval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
