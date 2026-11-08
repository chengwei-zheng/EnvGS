#!/bin/bash
# gui_f300_msurf_best.sh — opens the best flat-mirror result (2026-11-05, room scene B_framed / dataset_f300): learnable B-spline mirror
# surface + LiDAR first-echo depth, envgs_f300_msurf_spline_learn_ldepth005_dz5_fitfix (mirror SSIM / LPIPS-a / PSNR 0.973 / 0.022 / 38.64,
# full 39.82; baseline mgsn05 0.824 / 0.137 / 28.56, see mirror_lidar/doc/RESULTS.md). evc-gui on the display GPU (GPU2) with the scene-B viewer
# defaults. Closing the window does not save anything.
exec "$(dirname "$(readlink -f "$0")")/gui_mirror_b.sh" f300_msurf_spline_learn_ldepth005_dz5_fitfix
