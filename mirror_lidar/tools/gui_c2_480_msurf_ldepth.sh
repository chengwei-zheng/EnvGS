#!/bin/bash
# gui_c2_480_msurf_ldepth.sh — opens the corridor v2 convex-mirror result (2026-11-06, dataset_c2_480): learnable B-spline mirror surface
# + LiDAR first-echo depth, envgs_c2_480_msurf_spline_learn_ldepth005_dz5 (mirror SSIM / LPIPS-a / PSNR 0.831 / 0.150 / 27.89, full 34.12;
# B-spline only 0.801 / 0.192 / 26.17). Middle of the mirror ~33 dB, right side of the viewer image (u < -0.4 m) 24.7 dB: a Gaussian layer
# ~4 cm in front of the surface there (data/result/envgs/corridor/mirror_lr_c2_480_20261106). evc-gui on the display GPU (GPU2) with the c2_480 viewer defaults
# (configs/specs/corridor_c2_viewer.yaml, initial view = training camera 0093). Closing the window does not save anything.
GROUP=corridor VIEWER=corridor_c2_viewer exec "$(dirname "$(readlink -f "$0")")/gui_mirror_b.sh" c2_480_msurf_spline_learn_ldepth005_dz5
