"""2x2 box-downsample a raw export (RENDER / DEPTH .npy) into a new export dir: supersampling control (2026-10-30, c480).
    python downsample_raw.py <src_export_dir> <dst_export_dir> [factor=2]
"""
import glob
import os
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
f = int(sys.argv[3]) if len(sys.argv) > 3 else 2
for k in ('RENDER', 'DEPTH'):
    os.makedirs(os.path.join(dst, k), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, k, '*.npy'))):
        a = np.load(p)
        H, W = a.shape[0] // f * f, a.shape[1] // f * f
        a = a[:H, :W].reshape(H // f, f, W // f, f, -1).mean((1, 3))
        np.save(os.path.join(dst, k, os.path.basename(p)), a.astype(np.float32))
print('downsampled', src, '->', dst, a.shape)
