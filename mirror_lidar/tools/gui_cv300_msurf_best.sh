#!/bin/bash
# gui_cv300_msurf_best.sh — opens the best convex-mirror result (2026-11-05, room scene B_convex / dataset_cv300, R 1 m): learnable B-spline
# mirror surface + LiDAR first-echo depth, envgs_cv300_msurf_spline_learn_ldepth005_dz5 (mirror SSIM / LPIPS-a / PSNR 0.952 / 0.042 / 34.63,
# full 38.95; baseline 0.723 / 0.297 / 24.45, see mirror_lidar/doc/RESULTS.md). evc-gui on the display GPU (GPU2) with the scene-B viewer defaults.
# Closing the window does not save anything.
exec "$(dirname "$(readlink -f "$0")")/gui_mirror_b.sh" cv300_msurf_spline_learn_ldepth005_dz5
