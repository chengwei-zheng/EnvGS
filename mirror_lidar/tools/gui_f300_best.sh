#!/bin/bash
# gui_f300_best.sh — opens the best flat-mirror result: scene B_framed (f300), the default setting
# envgs_f300_lidarnorm_shared_bfilt_mgsn05 (mirror SSIM / LPIPS-a / PSNR 0.824 / 0.137 / 28.56, see mirror_lidar/doc/RESULTS.md),
# in evc-gui on the display GPU (GPU2) with the scene-B viewer defaults. Closing the window does not save anything.
exec "$(dirname "$(readlink -f "$0")")/gui_mirror_b.sh" f300_lidarnorm_shared_bfilt_mgsn05
