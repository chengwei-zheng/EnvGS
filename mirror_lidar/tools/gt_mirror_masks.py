# Oracle mirror masks from the Blender GT (2026-11-07, c2_480 oracle-mask test): mirror_masks/<cam>/000000.png in the format of
# lidar_mirror_mask.py (255 = mirror interior, 0 = rest; no boundary band), from persp/masks (== 3 = visible mirror), camera i <->
# frame i of persp/transforms.json. GT-based: for oracle tests only.
#     python gt_mirror_masks.py <blender persp dir> <out mirror_masks dir>
import json, os, sys
import cv2, numpy as np
P, out = sys.argv[1], sys.argv[2]
tj = json.load(open(os.path.join(P, 'transforms.json')))
n = []
for i, fr in enumerate(tj['frames']):
    g = cv2.imread(os.path.join(P, fr['mask_path']), cv2.IMREAD_UNCHANGED) == 3
    os.makedirs(os.path.join(out, f'{i:04d}'), exist_ok=True)
    cv2.imwrite(os.path.join(out, f'{i:04d}', '000000.png'), (g * 255).astype(np.uint8))
    n.append(g.mean())
n = np.array(n)
print(f'{len(n)} cameras -> {out}: {(n > 0).sum()} with mirror, mean mirror {100 * n[n > 0].mean():.1f}% of the image (frames with mirror)')
