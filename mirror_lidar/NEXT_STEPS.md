# 待做改进清单（2026-10-22）

训练流程和各项 loss 见 `TRAINING_RECIPE.md`。下面所有实验都在新数据 `dataset_r300_lamppanel` 上做（灯换成了发光网格面板，见下面第 0 步），以共用版为基准。

## 第 0 步：新数据和基准

1. **新数据**
   - 用 ~/SMBC/blender/project/proj_indoor_mirror_b_rot30/indoor_mirror_b_rot30_lamppanel.blend 渲染 `dataset_r300_lamppanel`，和 r300 是同一条轨迹。
   - 旧的 dataset_r300 有灯 bug：灯相机不可见、但镜中可见，镜面上出现过曝白块；共用版在这块上镜面 PSNR 被拉低约 1 dB。
2. **重新准备先验**
   - 转成 EasyVolcap 格式（COLMAP 用 GT 位姿三角化、compute_metadata）；
   - 算 StableNormal 法向；
   - 用 LiDAR 拟合镜面法向（lidar_mirror_normals.py）和镜面掩码（lidar_mirror_mask.py）；
   - 投影 LiDAR 第一回波深度（lidar_first_echo_depth.py）。
   - **2026-10-22 已完成**：EnvGS 场景名 `mirror_b/r300lp`（COLMAP 17.3 万点、重投影 0.374 px），`r300lp_lidar_depth`（每图 3.9 万点、对真值中位误差 1.6 cm），`r300lp_lidarnorm`（7×7 开运算最终版，法线误差 0.016°，掩码精度 99.96%、召回 96.8%）+ `mirror_masks`；脚本 `data/logs/prep_r300lp.sh`。数据集配置还没写。
3. **基准**：`lidarnorm_shared` 和 `lidarnorm_ldepth_shared`。
   - 评估：`eval_r300_lidar.sh`，加上镜面的 LPIPS / SSIM。镜面比较以这两项为主，PSNR 只作参考：共用版的镜中画面清晰，但有小的位置偏差，PSNR 吃亏。

## 改进项：每组在基准上只改一个因素

| # | 改进 | 状态 | 说明 |
|---|---|---|---|
| 1 | **只对 β 大的像素追踪反射** | 未实现，建议最先做 | 共用版每个像素都追全部约 50 万个面元，1 万次后只有原版速度的 1/3 到 1/2；加 LiDAR 深度后再慢 37%。先做这个，后面每组都能快很多。要定：阈值还是分位数（现有 `specular_filtering_percent` 是按分位数）、从第几次开始、reset 后 β 被压低时怎么保证还能涨回来（例如 reset 后暂停筛选一段，或保底追 LiDAR 镜面掩码内的像素）。等 betalog 的 β 恢复曲线出来再定。 |
| 2 | **β reset 从 clip 改成 ×0.1** | 代码已加（sampler `reset_specular_scale`） | clip 把 β 压到 0.001，排序信息全丢。注意 color sabotage（β ≤ 0.05 才打乱颜色）和 normal propagation（β ≥ 0.02 才放大）都依赖 β 阈值，换 reset 方式后这两者的行为也会变。对照组 r300_lidarnorm_shared_bsoft 在旧数据上跑到 2 万次就停了，没有结果。 |
| 3 | **LiDAR 点云初始化主面元** | 未实现 | r300 的 COLMAP 初始点有 8.7% 是镜中虚像点（`~/SMBC/EnvGS/data/vis/r300_init_points/`）。改用 LiDAR 第一回波点（世界坐标），颜色从 RGB 投影取；面元朝向用 LiDAR 法向或法向先验（现在是 torch.rand 随机）。对照组：COLMAP 去掉虚像点。 |
| 4 | **ldecay：LiDAR 深度权重 21000 次后 0.1 → 0.02** | 代码已加（`lidar_dpt_loss_weight_final` / `lidar_dpt_loss_decay_iter`） | 原版 ldepth 的 LiDAR loss 2000 次就降到 2.8 cm，最后 2.36 cm，基本是噪声底（σ 2.4 cm）；后期在拟合噪声，还拖累 RGB（不透明区域 PSNR −0.7 dB）。 |
| 5 | **nsplit：法向先验镜面 0.05、非镜面 0.01** | 代码已加（MirrorRegionSupervisor `mirror_norm_loss_weight`，镜面掩码自动生成） | 非镜面几何已经由 LiDAR 深度负责，不需要再加强 StableNormal（之前 nw5 把天花板搞差）。 |
| 6 | **mrgb：前 5000 次镜面 RGB / SSIM 权重 ×0.1** | 代码已加（`mirror_rgb_scale` / `mirror_rgb_until_iter`） | 反射分支第 3000 次才开始，之前镜中画面会被烤进漫反射。**依据要重新判断**：当时用 β 0.34 / 0.32 作证据，但 β 和 c_ref 可以此消彼长，β 不能读成反射比例；共用版的反射占比约 0.72，烤入问题可能已经小很多。 |

- 4–6 的旧配置（`envgs_r300_lidarnorm_ldepth_{ldecay,nsplit,mrgb}.yaml`）是原版（`shared_reflection: False`）、旧数据，10-21 启动十几分钟就停了，没有结果；重跑时要换成共用版和新数据。
- 顺序建议：第 0 步 → 1（提速）→ 2、3（和 β、初始化相关，可并行）→ 4、5、6（loss 权重）。

## β 恢复曲线（betalog，2026-10-22，旧数据 r300，共用版 + lidarnorm，跑到 2.66 万次）

图：`experiments/2026-10-22_r300_beta_recovery.png`。统计在训练视角上算：镜面 = LiDAR 掩码内部，非镜面 = 掩码 0 区域。

- **每次 reset 后恢复很快**：镜面 β 从 0.004 回升，+250 次约 0.1–0.16，+500 次约 0.5，+1000 次回到 reset 前的水平（0.55–0.63）。反射占比的走势相同。
- **18k 之后不再 reset**，镜面 β 从 0.55 稳步涨到 0.72，反射占比 0.74。
- **非镜面 β 一直在 0.0004–0.01 之间**，最后约 0.004，比镜面低两个数量级。按像素阈值筛选（比如 β > 0.02）能干净地分开镜面和非镜面。
- **镜面 c_ref 全程稳定在约 0.40**，reset 前后都不变。
- **每 1000 次有一个小波动**：镜面 β 掉 0.05–0.1，非镜面 β 冒一个尖峰，对应 color sabotage / normal propagation。
- **对第 1 项（只对 β 大的像素追踪）的含义**：
  - reset 后约 250 次内，镜面 β 也低于 0.02。如果直接按阈值筛，镜面就不追踪了；β 的梯度要靠追踪出来的 c_ref，于是 β 再也涨不回来。
  - 办法 a：reset 后暂停筛选约 1000 次（恢复期），只在 3k–18k 内全追 1/3 的时间。
  - 办法 b：配合 ×0.1 软 reset。reset 后镜面约 0.06、非镜面约 0.0004，排序保留，阈值按比例降低或按分位数筛。
  - 办法 c：LiDAR 镜面掩码内的像素保底追踪。

## 2026-10-22 20:1x 启动（新数据 r300lp，全部共用高斯，都带 β 日志 + 追踪像素占比 TRAIN/ref_frac）

| GPU | 运行 | 改动 |
|---|---|---|
| 0 | envgs_r300lp_lidarnorm_shared | 基准 |
| 1 | envgs_r300lp_lidarnorm_shared_bfilt | 第 1 项：只追踪 β > 0.02 的像素；反射开始（3000）和每次 β reset（6k–18k）后 1500 次内全部追踪。sampler 新参数 `specular_filtering_threshold`、`specular_filtering_pause_iter` |
| 3 | envgs_r300lp_lidarnorm_shared_bsoft | 第 2 项：β reset ×0.1 |

数据集配置 `configs/datasets/mirror_b/r300lp_lidarnorm.yaml`。

## 2026-10-23 结果：r300lp 三组（全部共用高斯；评估 `data/logs/eval_r300lp_lidar.sh` + 镜面 SSIM/LPIPS，38 个验证视角，其中 26 个有镜子）

| | 基准 shared | bfilt（β>0.02 才追踪） | bsoft（β ×0.1 reset） |
|---|---|---|---|
| 训练时间 | 20.6 h | **9.0 h** | 21.6 h |
| 后期追踪像素占比 | 1 | 0.135 | 1 |
| PSNR 全图 / 不透明 / 镜面 | 35.86 / 39.81 / 27.80 | 35.83 / **40.05** / 27.57 | 35.61 / 39.41 / 27.58 |
| SSIM / LPIPS 全图 | 0.9499 / 0.0635 | 0.9479 / **0.0615** | 0.9491 / 0.0650 |
| 镜面 SSIM / LPIPS-alex / LPIPS-vgg | **0.818** / 0.126 / 0.197 | 0.802 / **0.117** / **0.195** | 0.817 / 0.126 / 0.196 |
| 不透明深度：中位误差 / d<1.05 | 3.1 cm / 0.816 | **2.5 cm / 0.905** | 3.8 cm / 0.854 |
| 天花板深度中位误差 | 8.7 cm | **4.6 cm** | 8.3 cm |
| 镜面深度 std / 法线离散 p90 | 1 mm / 0.54° | 1 mm / 0.52° | 2 mm / 0.55° |
| 镜面 β / 反射占比 / c_ref | 0.761 / 0.765 / 0.386 | 0.757 / 0.763 / 0.388 | 0.752 / 0.761 / 0.391 |
| 非镜面 β | 0.0066 | **0.0026** | 0.0060 |

- 新数据（灯修好后）镜面 PSNR 27.80，旧 r300 共用版 26.25，+1.5 dB，就是之前灯的白块。
- **bfilt**：快 2.3 倍，画质基本不变（镜面 SSIM −0.016、PSNR −0.2，但 LPIPS 略好），不透明区域几何反而更好（非镜面像素不再受反射项干扰，β 也更低）。建议作为以后的默认设置。
- **bsoft**：reset 后镜面 β 从 0.14 而不是 0 开始回升，但 +500 次时只比 clip 高约 0.05，+1000 次两者一样；最终各项与基准持平或略差。软 reset 没有明显收益。
- 单次运行的波动约 0.2–0.3 dB，上面 PSNR 的差别都在波动范围内。
- β 曲线：`experiments/2026-10-23_r300lp_beta_recovery.png`。

## 2026-10-23 晚启动：bfilt + LiDAR 第一回波深度（每个 zone 按距离最近的回波，不用回波类型）

| GPU | 运行 | LiDAR 深度权重 |
|---|---|---|
| 0 | envgs_r300lp_lidarnorm_ldepth_shared_bfilt | 0.1，全程 |
| 1 | envgs_r300lp_lidarnorm_ldepth_ldecay_shared_bfilt | 0.1，21000 次起 0.02（第 4 项 ldecay） |

对照：envgs_r300lp_lidarnorm_shared_bfilt（无 LiDAR 深度）。

## 2026-10-23 第 3 项 LiDAR 点云初始化（GPU3：envgs_r300lp_lidarnorm_linit_shared_bfilt，对照 bfilt）

- 脚本 `mirror_lidar/tools/lidar_init_points.py` → `data/datasets/mirror_b/r300lp_lidarnorm_linit/sparse/0/points3D.ply`（其余是 r300lp_lidarnorm 的软链接）。
  - 262 个训练帧的第一回波点（1288 万），投到同帧相机取颜色（相机外和被遮挡的丢掉，剩 1025 万），4 cm 体素一个点。
  - 位置取 3×3×3 体素块内原始点的质心：测距噪声沿视线约 2.4 cm，单体素平均会偏离表面（2 cm 体素中位误差 2.3 cm），块质心能落回平面上。
  - 面元朝向仍然随机（torch.rand），只改了点的位置和颜色。
- 和 GT 表面比（GT 网格上 300 万采样点，间距约 1 cm）：

| | 点数 | 中位误差 | 离表面 > 5 cm | 镜中虚像点 | 房间外 |
|---|---|---|---|---|---|
| COLMAP（r300lp） | 172,570 | 0.61 cm | 11.92% | 14,231（8.25%） | 312 |
| LiDAR（4 cm） | 295,415 | 0.91 cm | 2.13% | 419（0.14%） | 0 |

- 可视化：`~/SMBC/EnvGS/data/vis/r300lp_init_points/{colmap,lidar}_init{,_marked}.ply`（marked：红 = 镜中虚像，橙 = 其它离群点，只保留房间内）。

### 10-24 results: LiDAR on top of bfilt (r300lp, 120 epochs, ~10.6-11 h each)

| run | PSNR | opaque PSNR | opaque depth med / d<1.05 | ceiling depth med | mirror SSIM / LPIPS-a / PSNR | opaque LPIPS-a | full LPIPS-a |
|---|---|---|---|---|---|---|---|
| bfilt | 35.83 | 40.05 | 2.5 cm / 0.905 | 4.6 cm | 0.802 / 0.117 / 28.13 | 0.051 | 0.062 |
| + ldepth (w 0.1) | 35.61 | 39.49 | 0.3 cm / 0.990 | 0.3 cm | 0.816 / 0.138 / 28.11 | 0.057 | 0.070 |
| + ldepth ldecay (0.1 -> 0.02 @21k) | 35.36 | 39.25 | 0.4 cm / 0.988 | 0.4 cm | 0.806 / 0.142 / 27.58 | 0.061 | 0.074 |
| + linit (LiDAR init points) | 35.80 | 40.03 | 1.0 cm / 0.941 | 6.2 cm | 0.814 / 0.140 / 27.93 | 0.040 | 0.055 |

- ldepth: depth almost exact everywhere (ceiling 4.6 -> 0.3 cm), costs ~0.5 dB opaque PSNR and opaque/mirror LPIPS.
- ldecay is worse than constant weight on every metric (decaying does not give back the RGB loss).
- linit: best opaque/full LPIPS & SSIM of all runs, walls fixed, but the ceiling stays wrong (no init help on textureless ceiling without depth loss).
- Mirror LPIPS: all three LiDAR variants ~0.14 vs bfilt 0.117, SSIM +0.01 - mixed; single-run noise.
- Next candidate: linit + ldepth.

### 10-24 launch: linit + ldepth, depth-weight sweep (bfilt, shared)
lidar_dpt_loss of ldepth plateaus at ~2.4 cm from 2k on, the L1 noise floor for sigma 2.4 cm is 1.9 cm (already at the floor);
RGB cost of ldepth: train img_loss 0.0074 vs 0.0071, ssim_loss 0.045 vs 0.041. Sweep the weight to see if the ceiling is still fixed at lower RGB cost.

| GPU | run | lidar_dpt_loss_weight |
|---|---|---|
| 0 | envgs_r300lp_lidarnorm_linit_ldepth_shared_bfilt | 0.1 |
| 1 | envgs_r300lp_lidarnorm_linit_ldepth_w03_shared_bfilt | 0.03 |
| 3 | envgs_r300lp_lidarnorm_linit_ldepth_w01_shared_bfilt | 0.01 |

### 10-24 results: linit + ldepth weight sweep

| run | PSNR | opaque PSNR | opaque depth med / d<1.05 | ceiling med | mirror SSIM / LPIPS-a / PSNR | opaque LPIPS-a | full LPIPS-a |
|---|---|---|---|---|---|---|---|
| bfilt | 35.83 | 40.05 | 2.5 cm / 0.905 | 4.6 cm | 0.802 / 0.117 / 28.13 | 0.051 | 0.062 |
| ldepth 0.1 | 35.61 | 39.49 | 0.3 cm / 0.990 | 0.3 cm | 0.816 / 0.138 / 28.11 | 0.057 | 0.070 |
| linit | 35.80 | 40.03 | 1.0 cm / 0.941 | 6.2 cm | 0.814 / 0.140 / 27.93 | 0.040 | 0.055 |
| linit + ldepth 0.1 | 35.27 | 39.59 | 0.3 cm / 0.989 | 0.3 cm | 0.785 / 0.155 / 27.09 | 0.047 | 0.064 |
| linit + ldepth 0.03 | 35.68 | 39.69 | 0.4 cm / 0.988 | 0.4 cm | 0.806 / 0.153 / 27.81 | 0.048 | 0.064 |
| linit + ldepth 0.01 | 35.45 | 39.67 | 0.5 cm / 0.986 | 0.8 cm | 0.807 / 0.152 / 27.56 | 0.044 | 0.060 |

- Even weight 0.01 fixes the ceiling (6.2 -> 0.8 cm); depth hardly depends on the weight in 0.01..0.1.
- linit + ldepth does not add up: opaque LPIPS between linit and ldepth, mirror LPIPS ~0.15 = worst of all runs (bfilt 0.117).
- Mirror LPIPS gets worse with every LiDAR addition (0.117 -> 0.14 -> 0.15): to check (per-frame / mirror crops) before more sweeps.

### 10-24 geometry vs GT mesh (mirror_lidar/tools/geom_eval_b.py, log data/logs/geom_eval_r300lp.log)
Fused point cloud of the rendered depth of the 38 val views vs GT mesh (acc / comp median cm, Chamfer = mean of the two mean
distances cm, F-score %); depth-derived normals pred vs GT (median deg); surfels with opacity > 0.5 (centre -> mesh, normal vs
nearest GT triangle).

| run | acc | comp | CD | F@1 | F@2 | F@5 | ceiling acc | mirror acc / F@1 | depth-normal opaque med / <10° | mirror normal med | surfels N / med / >5cm / virtual / normal med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shared (no bfilt) | 1.59 | 0.76 | 4.29 | 48.7 | 55.8 | 67.9 | 7.51 | 0.05 / 100.0 | 6.08 / 70.3 | 0.95 | 311k / 0.12 / 7.5% / 0.17% / 5.3° |
| bfilt | 1.60 | 0.78 | 3.27 | 47.7 | 58.2 | 78.4 | 3.62 | 0.06 / 100.0 | 5.70 / 72.3 | 0.95 | 314k / 0.12 / 4.8% / 0.16% / 5.3° |
| bsoft | 2.32 | 1.29 | 3.95 | 41.9 | 51.1 | 68.7 | 6.99 | 0.05 / 99.9 | 6.32 / 70.7 | 0.89 | 315k / 0.13 / 6.2% / 0.19% / 5.8° |
| ldepth 0.1 | **0.21** | **0.20** | **0.40** | **95.1** | **98.9** | **99.6** | **0.22** | 0.06 / 100.0 | **3.44 / 85.8** | 1.05 | 312k / 0.13 / 1.6% / 0.04% / 5.4° |
| ldepth ldecay | 0.30 | 0.24 | 0.58 | 86.9 | 96.0 | 99.4 | 0.37 | 0.07 / 100.0 | 3.71 / 84.0 | 0.97 | 299k / 0.13 / 1.8% / 0.03% / 5.3° |
| linit | 0.69 | 0.37 | 1.64 | 62.2 | 73.0 | 90.0 | 5.28 | 0.13 / 98.1 | 4.45 / 81.7 | 1.30 | 349k / 0.13 / 3.6% / 0.07% / 5.3° |
| linit + ldepth 0.1 | 0.22 | 0.20 | 0.41 | 95.1 | 98.7 | 99.5 | 0.25 | 0.10 / 99.5 | 3.45 / 86.0 | 1.05 | 345k / 0.13 / 1.3% / 0.03% / 5.5° |
| linit + ldepth 0.03 | 0.32 | 0.24 | 0.57 | 87.7 | 96.9 | 99.5 | 0.33 | 0.25 / 98.8 | 3.62 / 85.3 | 1.09 | 338k / 0.14 / 1.3% / 0.03% / 5.6° |
| linit + ldepth 0.01 | 0.38 | 0.27 | 0.74 | 80.6 | 92.2 | 99.2 | 0.59 | 0.22 / 98.7 | 3.83 / 83.8 | 1.21 | 349k / 0.14 / 1.3% / 0.03% / 5.5° |

- Geometry is monotonic in the depth weight; ldepth 0.1 alone = best; linit adds nothing on top of it.
- Opaque surfels (op > 0.5) sit on the surface in every run (median 0.12-0.14 cm): the depth error of bfilt / linit comes from
  blended low-opacity layers / floaters (> 5 cm surfels 4.8% bfilt, 1.3-1.6% with ldepth). Surfel normals ~5.3° everywhere.
- Mirror: the linit runs have a less exact mirror (acc 0.10-0.25 cm vs 0.06, offset -1..-3 mm toward the camera in mirror_flatness_b,
  normal 1.05-1.30° vs 0.95°) -> a lead for their worse mirror LPIPS.

### 10-25 which losses drive the mirror normal (mirror_lidar/tools/mirror_normal_grads.py, logs data/logs/mirror_normal_grads_*)
Trained checkpoints, training forward at iter 59000 on 8 mirror training views, each weighted loss backpropagated separately.
Restoring slope dL/d(delta) (1/rad, mean of the vertical / horizontal in-plane axis, (slope(+d) - slope(-d)) / 2, > 0 restoring),
mirror surfels tilted in place ('surfel') or the whole mirror rotated about its centre ('rigid'). bfilt, rigid:

| tilt | img (L1) | ssim | perc | norm prior | gs_norm | total |
|---|---|---|---|---|---|---|
| 2° | +0.118 | +0.189 | +0.034 | +0.017 | +0.001 | +0.359 |
| 5° | +0.077 | +0.010 | +0.002 | +0.016 | +0.001 | +0.107 |
| 10° | +0.043 | +0.032 | +0.003 | +0.012 | +0.001 | +0.091 |
| 20° | +0.009 | +0.032 | +0.002 | +0.013 | -0.000 | +0.056 |

- Near the optimum (2°) the RGB terms (img + ssim) are ~10x the normal prior; their pull drops with the tilt and is ~0 or
  negative (other runs) at 20° (narrow capture range, sign noise in ssim at 5-20°: views are random per call).
- The normal prior is a constant ~0.013-0.02 at every tilt (global); gs_norm ~0.001 (negligible); perc small.
- lidar_dpt (ldepth runs): ~0.018 for the rigid tilt at every angle (as strong as the normal prior), ~0 for the in-place tilt
  (it sees positions, not orientations).
- Rendered-normal gradient on mirror pixels at the trained state: |g_t| ssim 6 > img 2 > perc 1 >> norm 0.016 >> gs_norm 0.0003,
  i.e. the per-pixel RGB gradients are large but only partly coherent.
- Caveat: measured on converged models whose mirror image is already learned; early in training the RGB term has no correct
  target (reflection starts at 3000) - consistent with the r300 runs without prior collapsing into a virtual room.

## 10-26 删除镜前面元（不重训）测镜面 LPIPS

- 深度法向图上镜面有点状鼓包，LiDAR 组更多（`data/vis/r300lp_normals_f0/`）。统计：镜面圆盘内、镜子前表面（SM_Mirror 厚 8.6 mm，前表面在中面 +2.6 mm）前方 0.5–5 cm 的面元不透明度总和：bfilt 635，ldepth 2392，linit 3964，linit+ldepth0.03 5000。
- `mirror_lidar/tools/prune_mirror_front.py <run> [--dmin --dmax --tag]`：删掉这些面元，写到 `envgs_<run>_<tag>` 新目录（原模型不动），再跑 `data/logs/eval_prune.sh`、mirror_perc。

| 组 | 删掉面元数 | 镜面 SSIM | LPIPS-a | LPIPS-v | 镜面 PSNR |
|---|---|---|---|---|---|
| bfilt → pf05 | 6868 | 0.802 → 0.801 | 0.117 → 0.117 | 0.195 → 0.195 | 28.13 → 28.09 |
| ldepth → pf05 | 14940 | 0.816 → 0.815 | 0.138 → 0.136 | 0.208 → 0.205 | 28.11 → 28.04 |
| ldepth → pf02（0.2–5 cm） | 20286 | 0.816 → 0.800 | 0.138 → 0.153 | 0.208 → 0.229 | 28.11 → 27.35 |
| linit → pf05 | 22158 | 0.814 → 0.806 | 0.140 → 0.147 | 0.210 → 0.215 | 27.93 → 27.29 |
| linit+ldepth0.03 → pf05 | 27352 | 0.806 → 0.796 | 0.153 → 0.155 | 0.225 → 0.231 | 27.81 → 27.30 |

- 结论：直接删不能把 LPIPS 拉回 0.12；只有 ldepth 稍好（−0.002），linit 组反而变差。这些面元也承担了一部分颜色，删掉后镜子边缘出现缺块（`data/vis/r300lp_prune_mirror/crop_*.jpg`）。
- 裁剪图里差别主要在镜中物体本身（植物、椅子的形状和细节），不只在鼓包处。共用版的镜中画面就是场景面元的 DC 颜色，所以镜面 LPIPS 变差可能更多来自场景里被反射的物体（不透明区 LPIPS 在 ldepth 也变差：0.051 → 0.057）。鼓包是否有影响要靠重训验证（镜面区用平面深度替代 LiDAR 点）。

## 10-26 启动：去掉镜面鼓包的三组实验（`data/logs/launch_r300lp_bumps.sh`，13:52 起）

都以 `envgs_r300lp_lidarnorm_ldepth_shared_bfilt`（LiDAR 深度 0.1）为基准，结束后自动跑 `eval_r300lp_lidar.sh`：
- GPU3 `_ldepth_shared_bfilt_pf05_ft`：删掉镜前 0.5–5 cm 面元（`prune_mirror_front.py --pt`，同时删 Adam 状态）后接着训 5000 次（epochs 120→130，21000 后不再增密）；之后同一张卡跑 `_ldepth_shared_bfilt_ft`（不删，只多训 5000 次，对照）。
- GPU0 `_ldepth_mplane_shared_bfilt`：镜面掩码内的 LiDAR 深度样本换成 LiDAR 拟合平面的深度（`lidar_plane_depth.py` → `r300lp_lidar_depth_mplane`；镜面样本和 GT 的误差中位数从 1.80 cm 降到 0.021 cm；边界带样本删掉），完整 120 epoch。
- GPU1 `_ldepth_dz5_shared_bfilt`：LiDAR 深度 L1 加 5 cm 死区（`lidar_dpt_dead_zone: 0.05`，`max(|r| − 0.05, 0)`，约 2σ），完整 120 epoch。

## 10-26 微调结果（ldepth 0.1，接着训 5000 次）

| 组 | 镜面 SSIM | LPIPS-a | LPIPS-v | 镜面 PSNR | 全图 PSNR / LPIPS |
|---|---|---|---|---|---|
| ldepth（6 万次） | 0.816 | 0.138 | 0.208 | 28.11 | 35.61 / 0.0701 |
| 直接删（pf05，不训） | 0.815 | 0.136 | 0.205 | 28.04 | 35.59 / 0.0698 |
| 不删 + 5000 次（_ft，对照） | **0.837** | **0.124** | **0.191** | **28.97** | **35.94** / 0.0679 |
| 删 + 5000 次（_pf05_ft） | 0.802 | 0.136 | 0.210 | 27.75 | 35.48 / 0.0700 |

- 删镜前面元有害：同样多训 5000 次，删了的比不删的镜面 LPIPS 差 0.012、PSNR 差 1.2 dB（21000 后不增密，删掉的部分补不回来）。
- 意外：只是多训 5000 次，镜面 LPIPS 就从 0.138 降到 0.124、PSNR +0.9 dB。原训练 VAL PSNR 在 55k→60k 还涨 0.3 dB，没收敛；xyz lr 已在终值 3e-6，其它 lr 恒定。镜面 LPIPS 和 bfilt 的差距可能部分是收敛速度问题 → 加跑 `_shared_bfilt_ft`（bfilt 同样多训 5000 次）看差距是否还在。
- bfilt 同样多训 5000 次（`_shared_bfilt_ft`）反而变差：镜面 SSIM 0.802 → 0.797，LPIPS-a 0.117 → 0.127，LPIPS-v 0.195 → 0.206，镜面 PSNR 28.13 → 27.39；全图 PSNR 35.83 → 35.53。
- **结论修正**：训练末段（学习率已到终值）镜面指标随迭代来回波动，幅度约 LPIPS ±0.01、镜面 PSNR ±0.8 dB。只看单个最终 checkpoint 时，bfilt 和 ldepth 的镜面 LPIPS 差距（0.117 vs 0.138），以及删 / 不删微调之间的差距（0.136 vs 0.124），都在这个波动范围内，不能算定论。以后镜面指标要取末段多个 checkpoint 的平均（或多 seed）。

## 10-27 结果：镜面平面深度（mplane）/ 死区（dz5）

| 组 | 全图 PSNR / LPIPS | 镜面 SSIM | LPIPS-a | LPIPS-v | 镜面 PSNR | 不透明区 LPIPS-a | CD (cm) | F@1 | 天花板 (cm) | 镜面深度法向中位 / <2° | 镜前 0.5–5 cm Σ不透明度 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bfilt | 35.83 / 0.0615 | 0.802 | 0.117 | 0.195 | 28.13 | 0.051 | 3.27 | 47.7 | 3.62 | 0.95° / 74.6% | 626 |
| ldepth 0.1 | 35.61 / 0.0701 | 0.816 | 0.138 | 0.208 | 28.11 | 0.057 | 0.40 | 95.1 | 0.22 | 1.05° / 71.4% | 2353 |
| ldepth_mplane | 35.31 / 0.0729 | 0.805 | 0.135 | 0.213 | 27.82 | 0.061 | 0.42 | 94.7 | 0.25 | **0.61° / 81.6%** | 2197 |
| ldepth_dz5 | 35.64 / 0.0680 | **0.832** | 0.123 | 0.192 | 28.36 | 0.057 | 0.59 | 84.9 | 0.48 | 0.96° / 72.3% | 2402 |

- mplane：镜面深度法向明显变好（0.61°，比不加 LiDAR 的 bfilt 还好），几何其它项和 ldepth 持平；但画质没改善（全图 PSNR −0.3 dB）。
- dz5：画质在 LiDAR 组里最好（镜面 LPIPS 0.123、全图 LPIPS 0.068），几何比 ldepth 差一些（F@1 85%，天花板 0.48 cm，仍远好于 bfilt）。
- 两组镜前面元的不透明度总和都没降（约 2200–2400）；深度法向图（`data/vis/r300lp_normals_f0_bumps/`）上鼓包变小变散，但仍在，bfilt 也有少量。说明鼓包主要不是镜面上 LiDAR 噪声造成的。
- 镜面指标末段波动约 LPIPS ±0.01（10-26 微调实验），以上镜面 LPIPS 差异大多在这个范围内。
- 镜面 gs_norm（渲染法向 vs 深度法向的 1−cos，PNG 解码，镜面内缩 9 px，不乘深度系数）：基准 0.00220、bfilt 0.00218、ldepth 0.00351、mplane 0.00244、dz5 0.00345、linit 0.00592、linit+ldepth0.03 0.00672；夹角 >5° 的像素占比 5.7 / 7.4 / 4.6 / 7.2 / 13.3 / 11.3%（bfilt / ldepth / mplane / dz5 / linit / linit+ldepth0.03）。不透明区约 0.031–0.040，所以镜面鼓包只占全图 gs_norm 的 1–2%，再乘权重 0.04，几乎没有梯度。用户在 viewer 里看，镜面区域 bfilt 最好、鼓包影响明显。可选：镜面掩码内给 gs_norm 单独加权，或打开 gs_dist（2DGS distortion，目前权重 0）。

## 10-27 启动：鼓包第二轮（`data/logs/launch_r300lp_bumps2.sh`），都以 bfilt 为基准

- mplane 为什么也有鼓包：`dpt_map` 是加权平均深度（depth_ratio 0），LiDAR L1 只约束平均值。镜面像素的渲染深度 mplane 最好（镜前 >2 mm 1.7%，bfilt 3.2%），但镜面附近面元的不透明度总和是 bfilt 的 2.5 倍（8911 vs 3565），41% 在镜前 >2 mm、25% 在玻璃内，前后两层平均出正确深度。
- GPU0 `_shared_bfilt_dist100`：+ gs_dist（2DGS distortion，已验证实现正确，`check_dist_loss.py`）权重 100，从 3000 次起。
- GPU1 `_shared_bfilt_mgsn1`：+ 镜面掩码内部额外 gs_norm（`mirror_gs_norm_loss_weight: 1.0`，不乘深度系数；MirrorRegionSupervisor 新加）。
- GPU3 `_ldepth_nomirror_shared_bfilt`：+ LiDAR 深度 0.1，但镜面掩码内部和边缘一圈的样本全部去掉（`lidar_plane_depth.py --drop_mirror` → `r300lp_lidar_depth_nomirror`）。
- 每组训完自动跑：GT 评估、`mirror_perc_b.py`（镜面 SSIM/LPIPS，从 scratchpad 移进 tools）、`geom_eval_b.py`、`mirror_bumps_b.py`（新：镜面 gs_norm 和夹角 >2/5/10° 占比）。
- 现有组的鼓包指标（镜面 1−cos / 夹角中位数 / >5° / >10°）：bfilt 0.00218 / 0.97° / 5.7% / 1.5%；ldepth 0.00351 / 1.00° / 7.4% / 2.3%；mplane 0.00244 / 0.63° / 4.6% / 1.5%。
- 11:47 `_dist100` 停掉（约 7k 次）：gs_dist 从 3000 次开始后，训练 PSNR 一直比 bfilt 低 2–5 dB（3000–3500：27.1 vs 25.5；6500–7000：30.9 vs 27.1），差距没有缩小，权重太强。GPU0 改跑 `_shared_bfilt_dist10`（权重 10，`launch_r300lp_dist10.sh`，评估流程相同）。`_dist100` 的 trained_model / record 目录保留，配置首行标 ABORTED。
- 约 7k 次时另外两组正常：mgsn1 训练 PSNR 29.2（bfilt 29.7），镜面 gs_norm 项 3.3e-4；nomirror 29.4。
- dist10 前 9000 次：训练 PSNR 比 bfilt 低 0.3–0.8 dB（dist100 是 2–5 dB），gs_dist 项约 3.5e-5–9e-5，正常继续。
- 10-27 21:18 结果（mirror 指标 SSIM/LPIPS-a/PSNR；鼓包 1-cos / 中位角 / >5° / >10°）：
  - dist10：全图 PSNR 34.02 / LPIPS 0.0810（bfilt 35.83 / 0.0615），不透明区 PSNR 35.04（bfilt 40.05）、天花板 5.56 cm（bfilt 3.62）；镜面 0.867 / 0.119 / 30.09（目前最好，bfilt 0.802 / 0.117 / 28.13）；鼓包 0.00256 / 0.78° / 6.1% / 1.8%（bfilt 0.00218 / 0.97° / 5.7% / 1.5%），没去掉鼓包，伤了不透明区。
  - nomirror（ldepth0.1、镜面不加 LiDAR）：全图 34.95 / 0.0727，不透明区几何和 ldepth 一样好（F@1 95.7、天花板 0.21 cm）；但镜面 0.782 / 0.192 / 26.49、镜面深度 acc 1.41 cm、F@1 30.8，鼓包 0.00855 / 2.05° / 20.8% / 8.1%，全组最差 —— 镜面没 LiDAR 约束时镜面面元被周围 LiDAR 拉坏。
  - nomirror 原因（22:00 查）：镜面整体往外鼓成穹顶（面元相对前表面平均偏移：中心 +1.8 cm、0.9r 处 +0.15 cm；bfilt 约 −1 cm，在玻璃内），法线仍平 → 阶梯状 → 鼓包；第 79 epoch 已如此；镜面面元不透明度总和 21540（bfilt 6615）。推测：房间被 LiDAR 固定后，画面误差偏好镜面前移 1.5–2 cm，镜面有 LiDAR（ldepth/mplane）时两头拉扯成前后两层 = 鼓包，没有 LiDAR 时整体前移。待验证：把镜面面元整体平移 −3..+3 cm 看镜面画面误差最小处。
  - mgsn1（镜面内 gs_norm 权重 1.0）：鼓包基本消失 0.00085 / 0.45° / 0.25% / 0.12%，镜面面元 54% 在 ±0.2 cm 内（bfilt 22%），镜面 0.832 / 0.132 / 29.06；但全图 33.80 / 0.1138、不透明区 35.18、墙 −4.4～−5 dB、不透明 F@1 30.4、天花板 6.89 cm。dist10 不透明区也掉 5 dB（35.04），训练 PSNR 末 10% 都比 bfilt 低约 2 dB（33.9 / 33.8 vs 35.8）—— 原因未查明（正则副作用还是 bfilt 单次运行偏好，需要重跑 bfilt 当对照）。
  - 法线对比图：data/vis/r300lp_round2_surfnorm/。
  - mgsn1 镜边 floater + 镜后墙面模糊是同一件事（22:50 查）：镜面平面上的面元（偏移 ±2 cm、法线与镜面一致）伸出镜子边缘，半径 1.00–1.10 r（r = 0.54 m，即镜外 0–5.4 cm）不透明度总和 5546，bfilt 只有 82；dist10 伸到 1.3 r。误差变大的像素（如 cam 280/48/0，图像 1.2–1.8%）渲染深度比 GT 近 0.3–2 m，反投影正落在镜面平面、半径 1.0–1.07 r —— 斜着看时这圈伸出来的镜面盖在镜后墙面上。镜面掩码本身没问题（255 区在 GT 镜面外的只有 0.03%，离镜边约 11 px）。推测：镜内深度-法线一致（及 dist）惩罚镜边内侧"镜面半透明、混入后方墙面深度"的像素，最省事的解是把平整、不透明的镜面面元往外扩；镜外只有画面误差在拦。对比图 data/vis/r300lp_mgsn1_diag/。
- 10-27 22:47 第三轮（用户：镜外约束不行，因为镜子可能有镜框）：
  - `_mgsn1_dg`（GPU0）：镜面 gs_norm 1.0，只在 dist_map < 1e-6 的像素上加（新参数 mirror_gs_norm_dist_max；累计 alpha 不能用，封闭房间处处≈1）。
  - `_mgsn02`（GPU1）：镜面 gs_norm 0.2，从 3000 次开始（新参数 mirror_gs_norm_start_iter）。启动脚本 data/logs/launch_r300lp_mgsn2.sh。
  - 新场景 B_framed（镜子加厚到 3 cm + 木框）：~/SMBC/blender/project/proj_indoor_mirror_b_framed/，等用户看过 GUI 再在 GPU3 上渲染。
  - 23:20 门槛检查：_mgsn1_dg 训练中只有 11–40% 镜面像素通过 dist<1e-6（6k 次时 11%）。离线用光栅器渲染 dist（scratchpad nm/dist_offline.py，38 个视角里有镜子的）：
    bfilt 镜内 p50 1.1e-6 / p90 7.5e-6；镜边内侧 11 px p50 4.3e-7（80% < 1e-6，**并不混层**）；镜外 11 px 一圈 p50 7.3e-4（镜边 + 后方墙混层）。
    mgsn1 镜内 p50 2.0e-7，镜外一圈 p50 2.3e-6（伸出去的镜面把这圈变成单层）。
    → "镜边内侧混进墙面深度"的假设不成立，dist 门槛没有意义；外扩的机制还没找到。
  - 23:35 用户选：停 _mgsn1_dg（约 7k 次，配置首行标 ABORTED），GPU0 改跑 `_mgsn1_er30`：镜面 gs_norm 1.0，只加在 LiDAR 镜内掩码再腐蚀 30 px（全分辨率，离镜边约 41 px）的区域，新参数 mirror_gs_norm_erode_px；保留的镜内像素约 78%。启动脚本 data/logs/launch_r300lp_er30.sh。
  - 新场景 dataset_f300（B_framed）23:11 起在 GPU3 渲染：~/SMBC/blender/project/proj_indoor_mirror_b_framed/f300_run.log。
- 10-28 **用户选定当前最好设置 = `envgs_r300lp_lidarnorm_shared_bfilt_mgsn02`**（bfilt + 镜面掩码内额外 gs_norm 0.2、从 3000 次开始；"镜面视觉效果目前最好、背景不算太糊"）。指标：全图 34.79 / 0.0848，不透明区 38.23，镜面 0.800 / 0.153 / 27.64，鼓包 >5° 1.5% / >10° 0.29%，镜外 0–1.6 cm 有镜面面元（mgsn1 是 0–5 cm）。
  - 接着跑两组：
    - `envgs_r300lp_lidarnorm_ldepth_dz5_shared_bfilt_mgsn02`（GPU1，09:2x 起）：+ LiDAR 第一回波深度 0.1 + 5 cm 死区。选 dz5 的理由：bfilt 上几种 LiDAR 版本里 dz5 的画质最好（全图 LPIPS 0.068、镜面 0.832 / 0.123），死区让镜面上不跟测距噪声拉扯；注意 bfilt 上加 LiDAR 从没提高过不透明区 PSNR（40.05 → 39.3–39.5），主要提升几何。
    - `envgs_f300_lidarnorm_shared_bfilt_mgsn02`（GPU3）：新场景 B_framed。数据准备 data/logs/prep_f300.sh（09:28 起），完成后用 data/logs/launch_f300_mgsn02.sh 训练 + 用 dataset_f300 真值评估（eval_f300_lidar.sh）。

- 10-28 10:50 f300 预处理完成(COLMAP 重投影 0.36 px;LiDAR 平面法线误差 0.023°;掩码内部精度 99.99%/召回 92.7%,IoU 中位 0.979),
  配置 configs/datasets/mirror_b/f300_lidarnorm.yaml + envgs_f300_lidarnorm_shared_bfilt_mgsn02.yaml,GPU3 开训(launch_f300_mgsn02.sh)。
- 10-28 13:50 mgsn1_er30(权重 1.0,掩码内圈腐蚀 30 px)失败:全图 33.27/0.1153,不透明 35.36,镜面 SSIM/LPIPS 0.779/0.167,
  鼓包 >5° 1.66%/>10° 0.75%,墙 SM_Wall 34.43 / Wall.001 36.50(和 mgsn1 一样掉)。说明墙变差不是只来自镜边外扩,
  而是权重 1.0 本身;镜边留 30 px 不约束反而让镜面变差。结论:保持 mgsn02。

## 10-28 晚 / 10-29 结果：dz5 + mgsn02（r300lp）、B_framed（f300）权重扫描

指标顺序：全图 PSNR / LPIPS；不透明区 PSNR；镜面 SSIM / LPIPS-a / PSNR；鼓包 >5° / >10°；墙 SM_Wall / SM_Wall.001 PSNR；几何（整体 acc cm / F@2 %，天花板 acc cm）。

- r300lp `_ldepth_dz5_shared_bfilt_mgsn02`：35.24 / 0.0812；38.55；0.804 / 0.142 / 28.18；2.00% / 0.47%；37.82 / 38.98；0.35 / 93.9，天花板 0.51。
  对照 `_mgsn02`：34.79 / 0.0848；38.23；0.800 / 0.153 / 27.64；1.54% / 0.29%；37.25 / 39.42；1.33 / 58.0，天花板 6.57。
  → 加 LiDAR 深度几乎全面更好（几何大幅提升），只有鼓包略多。
- f300（B_framed）三组，都是 `envgs_f300_lidarnorm_shared_bfilt_mgsnXX`（启动 data/logs/launch_f300_mgsn02.sh、launch_f300_sweep.sh）：

  | 镜面 gs_norm | 全图 | 不透明区 PSNR / LPIPS | 镜面 | 鼓包 >5° / >10° | 墙 | 镜面深度 MAE | 天花板 acc |
  |---|---|---|---|---|---|---|---|
  | 0.1 | 35.46 / 0.0730 | 40.35 / 0.060 | 0.778 / 0.140 / 26.86 | 14.2% / 1.57% | 39.12 / 39.89 | 0.6 cm | 6.9 cm |
  | 0.2 | 35.06 / 0.0778 | 40.28 / 0.062 | 0.761 / 0.169 / 26.10 | 4.28% / 0.95% | 39.17 / 40.07 | 2.2 cm | 5.2 cm |
  | **0.5** | 36.00 / 0.0847 | 39.68 / 0.074 | **0.824 / 0.137 / 28.56** | **2.75% / 0.63%** | 38.59 / 38.50 | 0.7 cm | 9.2 cm |

  - 0.2 的镜面 LPIPS 是三组最差、不单调，且只有它镜面深度偏 2 cm（之前猜"贴到镜框前沿"，0.1/0.5 都只偏 0.6–0.7 cm，所以更可能是单次运行的随机结果）。
  - 0.5 代价：不透明区 LPIPS +0.012、墙 −0.6 / −1.5 dB、天花板几何变差；和 r300lp 上 1.0 的现象同向但轻得多。
- **10-29 用户决定：之后在 B_framed（f300）上做，`envgs_f300_lidarnorm_shared_bfilt_mgsn05` 是这个场景当前的默认设置**（viewer 里镜面效果好）。
- 还在跑：`envgs_f300_lidarnorm_ldepth_dz5_shared_bfilt_mgsn02`（GPU3）。若 dz5 能把墙/天花板拉回来，下一组试 dz5 + mgsn05。
- 10-29 13:xx `envgs_f300_lidarnorm_ldepth_dz5_shared_bfilt_mgsn02`：35.34 / 0.0818；不透明区 39.45；镜面 0.799 / 0.140 / 27.53；鼓包 >5° 7.57% / >10° 1.03%；
  墙 38.31 / 38.91；几何 acc 0.39 cm、F@2 94.7%、天花板 0.42 cm。→ LiDAR 深度在 f300 上同样只改几何，墙/不透明区画质没拉回（反而 −0.8 dB）。

## 10-29 走廊凸面镜场景 c480：数据和 LiDAR 曲面镜先验
- 数据：`data/logs/prep_c480.sh`（COLMAP 重投影 0.75 px，LiDAR 第一回波深度中位误差 1.63 cm，StableNormal）→ `data/datasets/corridor/c480`。
- 曲面镜先验：新工具 `mirror_lidar/tools/lidar_mirror_normals_curved.py`（无平面假设：平滑度检查 + 体素聚类 + 镜面 (u,v) 1 cm 格子平均回波 +
  `quadric_mls` 自由二次曲面 MLS，h 0.2、τ 0.3；高度场求交带收敛检查；范围 = 1/4 密度区域凸包）+ `lidar_mirror_mask.py`，脚本 `data/logs/prep_c480_mirror.sh`。
  GT 球面（从 ConvexMirror 顶点拟合）：中心 (−0.02669, 2.42669, 2.25706)，R 1.0，半张角 36.87°。
- 结果（v2）：曲面 vs GT 法线 内部中位 0.28° / p95 2.0°、外 5 cm 2.2° / 4.8°，位置 p95 3.4 mm；κ (1.07, 0.99)。
  掩码：精度 100.00%、召回 86.4%、IoU 中位 0.875；涂上的法线误差中位 0.275°（最差帧 p99 4.7°）。拟合范围比真实镜面小一圈（投影 0.98 vs 1.13 m²）。
- v1（`c480_lidarnorm_v1`，保留作对照）两个 bug：高度场求交没有收敛检查 → 墙上误涂（精度 89.7%）；半密度阈值切掉轨迹少看的一侧（召回 74%）。
- 关键发现：仿真测距噪声 2.6 cm 时直接对回波做代数 MLS 会把曲率高估到 1.6–3.6（Taubin 也一样），先按曲面格子平均回波才消除，见
  experiments/2026-10-28_quadric_mls_synth/README.md。
- 配置已备好：`configs/datasets/corridor/{corridor,c480_lidarnorm}.yaml`、`configs/exps/envgs/corridor/envgs_c480_lidarnorm_shared_bfilt_mgsn05.yaml`（未开训）。
  评估脚本还是场景 B 专用（mirror_b 路径、geom_eval 写死平面镜、平整度/法线分散按平面），训练前后要改。
- **10-29 16:19 c480 开训**（`data/logs/launch_c480.sh`，用户：看曲面镜是否需要深度监督）：都是 f300 默认 _mgsn05 设置
  - GPU0 `envgs_c480_lidarnorm_shared_bfilt_mgsn05`（无 LiDAR 深度）
  - GPU1 `envgs_c480_lidarnorm_ldepth005_dz5_shared_bfilt_mgsn05`（LiDAR 第一回波深度 0.05，5 cm 死区）
  - GPU3 `envgs_c480_lidarnorm_ldepth001_dz5_shared_bfilt_mgsn05`（0.01，5 cm 死区）
  - 训完自动跑 `data/logs/eval_c480.sh`：GT 评估、mirror_perc / geom_eval / mirror_bumps（加了 RESULT_GROUP、CEILING、MIRROR_SPHERE 环境变量，
    默认值不变）+ 新的 `mirror_normal_sphere.py`（渲染法线 / 深度法线 vs GT 球面法线；NORMAL 图是 OpenGL 相机系，f300 上验证：
    mgsn05 深度法线 >5° 2.7%，和 mirror_bumps 的 2.75% 一致）。
- **10-30 c480 结果**（60 val 帧，35 帧有镜；曲面镜法线 = `mirror_normal_sphere.py` 对 GT 球面，腐蚀 9 px）：

  | 深度权重 | PSNR 全/不透明/镜 | 镜 SSIM / LPIPS-a | 全 LPIPS | 深度 MAE 不透明/镜 (cm) | 融合点 CD / F@2 | 镜 acc / F@2 | 镜 NORMAL 中位 / >5° | 镜 SURFACE 中位 / >5° | bumps >10° |
  |---|---|---|---|---|---|---|---|---|---|
  | 0 | 32.56 / 33.26 / 23.74 | 0.7125 / 0.3065 | 0.1314 | 4.6 / 4.6 | 2.78 / 44.2 | 3.67 / 16.2 | 2.51° / 16.2% | 3.96° / 35.8% | 8.8% |
  | 0.01 | 32.45 / 33.26 / 23.25 | 0.7032 / 0.3316 | 0.1310 | 3.4 / 3.5 | 2.09 / 55.0 | 2.21 / 46.4 | 2.49° / 17.5% | 3.39° / 28.8% | 8.3% |
  | 0.05 | 32.15 / 32.86 / 23.15 | 0.7010 / 0.3441 | 0.1411 | 2.5 / 3.5 | 1.36 / 78.1 | 1.70 / 87.4 | 2.81° / 22.5% | 3.55° / 32.2% | 12.0% |

  结论：深度监督让几何（含镜面位置、虚像面元 2.2% → 0.2%）明显变好，但镜面法线（NORMAL）和镜中画质不变好甚至变差；0.01 几乎不伤画质、
  SURFACE_NORMAL 最好，是折中；曲面镜法线主要靠 LiDAR 法线先验，不靠深度。镜内画质三组都只有 PSNR 23–24 / LPIPS 0.31–0.34，瓶颈不在几何。
- **10-30 曲面镜差的原因：β 没起来，反射被烘焙成镜面面元颜色**。镜面 GT 像素上 β 中位 0.016（f300 0.741），反射占比约 0；
  训练曲线 mirror_beta 全程 0.03–0.08（f300 在 3000 开反射后 750 步内到 0.4）。替换法线对照（`tools/export_oracle_sphere.py`、
  `data/logs/oracle_c480.sh`，只改推理时镜面像素的反射光线，结果在 `envgs_<run>_oracle_<tag>/`），无深度组镜面 PSNR / SSIM / LPIPS-a：
  原始 23.74 / 0.713 / 0.307；GT 法线 23.71（β 太小，几乎不变）；GT 法线+GT 起点 23.83；强制 β=0.9 + GT 法线和起点 23.36 / 0.654 / 0.305；
  β=1.0 22.79；β=1.0 + 模型自己的法线 18.86 / 0.323 / 0.473。
  → 模型法线如果真用来反射会差 4 dB；但就算几何全对，靠现有基础高斯反射也只到 ~23，不比烘焙好，所以优化停在烘焙解。
  凸面镜虚像都在镜后 0.5 m 内、视差小，烘焙贴图很容易解释图像。待查：反射内容差是走样（超采样对照）还是镜中看到的区域重建差；
  训练上要在 LiDAR 镜面掩码上把 β 拉起来（β 先验/早期强制）并用 LiDAR 曲面法线。
- **10-30 超采样对照**（`SS=2 data/logs/oracle_c480.sh`：2592x1944 渲染后 2x2 盒式下采样，`tools/downsample_raw.py`；无深度组，镜面 PSNR / SSIM / LPIPS-a）：
  烘焙原样 1x 23.74 / 0.713 / 0.307 → 2x 23.63 / 0.712 / 0.316（不变）；GT 几何 + β=0.9 反射 1x 23.36 / 0.654 / 0.305 → 2x **24.28 / 0.751 / 0.202**；
  β=1.0 2x 23.85 / 0.733 / 0.210。→ 走样是反射路径的主要损失之一，超采样后反射已明显好过烘焙（LPIPS 0.20 vs 0.31），
  但离 f300 平面镜（28.56 / 0.824 / 0.137）还差很多，剩余差距待分（更高超采样 / 镜中区域重建质量 / 镜面反射率）。
- **10-30 反射多采样（sampler `ref_ss`）**：每个追踪像素在像素内 √K×√K 分层位置（训练时格内抖动）各发一条反射光线，法线 / 表面点 / 相机光线
  从渲染图双线性插值，颜色取平均；只在 specular 过滤生效时用（全图追踪阶段仍 1 条，否则 5 s/迭代）。现有模型 + GT 球面几何 + β 0.9 验证：
  K=4 24.27 / 0.748 / 0.204（≈ 2× 超采样 24.28 / 0.751 / 0.202），K=16 **24.46 / 0.782 / 0.172**；不替换（烘焙原样）K=4 不变。
- **10-30 13:05 c480 分步改进开训**（`data/logs/launch_c480_fix.sh`，用户：一步步加）：
  - GPU0 `envgs_c480_gtnorm_shared_bfilt_mgsn05`：1) 镜面 GT 球面法线先验 + GT 镜面掩码（`c480_gtnorm`，`tools/make_gtnorm_sphere.py`；和 LiDAR 法线差中位 0.45°）
  - GPU1 `..._mbeta`：2) 1 + 镜面 β 损失（MirrorRegionSupervisor `mirror_beta_loss_weight` 0.1，relu(0.9 − β) 按镜面像素平均，3000 起）
  - GPU3 `..._mbeta_rss4`：3) 1 + 2 + `ref_ss: 4`

## 11-01 c480 分步结果（GT 法线 / + β 损失 / + 多采样）
- 1）`c480_gtnorm_shared_bfilt_mgsn05`：全图 32.71 / 0.902 / 0.122；不透明区 33.56；镜面 23.89 / 0.721 / 0.289（原 23.74 / 0.713 / 0.307）；
  镜面法线 vs GT 球面中位 2.06°、>5° 8.5%（原 2.51° / 16.2%）；镜面深度 MAE 2.2 cm（原 4.6）；鼓包 >10° 3.6%（原 8.8%）；1.61M 高斯（不再顶上限）。
  → 几何/法线明显好，但镜面 β 全程 0.03–0.08，反射仍烘焙进面元颜色，镜面画质几乎不变。法线准不是瓶颈。
- 2）`_mbeta`：**崩了**。全图 29.84 / 0.808 / 0.271，镜面 18.80 / 0.512 / 0.742，镜面深度偏 70 cm、法线中位 26°。训练 PSNR 正常（~33），
  但正对镜子的验证视角（cam 0072–0168、0328–0368）满屏是相机附近的大块针状浮游高斯（β 高、纯反射）。验证曲线：cam0360 从 5000 步
  起就比 1）低 2.4 dB，2 万步左右（高斯数冲到 1.8M 上限）再掉到 18 dB；典型过拟合。逐像素 β 损失会把挡在镜面像素前的任何高斯
  （含浮游物）的 β 拉高，β 高的浮游物用追踪到的反射拟合各训练视角。
- 3）`_mbeta_rss4`：训练中途（44999 步）验证同样崩（cam0360 18.1，全图 30.56），同一失败模式。
- 下一步候选：β 损失只作用于 3D 上靠近镜面的高斯（按 GT 球面 / LiDAR 镜面距离挑高斯，逐高斯而非逐像素），或按像素的渲染深度
  与镜面深度一致才加；同时压浮游物（opacity / scale 正则、相机近处剔除）。
- 11-01 用户猜想"平面镜场景看得到镜后、走廊看不到"：统计镜后 0.1/0.3/0.5/1.0 m 的圆盘采样点被多少视角直接看到（不经镜面、未被遮挡；
  脚本在会话草稿 behind_mirror.py）：f300 83/108/117/136 of 300 视角、采样点 100% 被看到；b300 141/128/133/131、100%；
  c480 20/0/62/121 of 480、15%/0%/3%/5%。→ 走廊凸面镜后面基本没被直接看到，猜想的前提成立。
  但：r300 和 f300 同一条轨迹（同样看得到镜后），基线仍退化成虚像房间，只有 gtnorm 对；且 c480 的失败不是虚像房间（镜面深度对），
  而是反射烘焙进镜面颜色、以及 β 损失后的相机近处浮游物——这两种都不靠"镜后有东西"，看到镜后也排除不了。
- **11-01 新场景 B_convex（cv300）**：用户提议在平面镜训得好的 B_framed 里只把镜面换成 R 1 m 凸面镜，分开"曲面本身"和"走廊场景"两种原因。
  ~/SMBC/blender/project/proj_indoor_mirror_b_convex/（build_mirror_b_convex.py；轮缘留原位、中心凸出 15.9 cm、口径半径 0.541 m、
  半张角 32.8°；球心等存 mirror_sphere.json；相机离镜最近 0.45 m，14 帧 < 0.5 m LiDAR 盲区）。轨迹/划分同 f300。
  一键流程 data/logs/pipeline_cv300.sh（渲染 GPU 0/1/3 → 预处理 → 曲面 LiDAR 法线 cv300_lidarnorm / GT 球面法线 cv300_gtnorm →
  训 envgs_cv300_{lidarnorm,gtnorm}_shared_bfilt_mgsn05（f300 默认）GPU 0/1 → eval_cv300.sh），11-01 10:54 起，日志 pipeline_cv300.log。

## 2026-11-02 cv300 结果(B_framed 只把镜子换成 R 1 m 凸面镜,f300 默认设置 mgsn05,不加 β loss)
| run | 镜面 PSNR / SSIM / LPIPS-a | mirror_beta | 反射占比 | 法线 vs GT 球 med / p90 | 镜面深度 MAE |
|---|---|---|---|---|---|
| cv300_lidarnorm | 24.45 / 0.723 / 0.297 | 0.62 | 0.61 | 1.39° / 4.17° | 1.4 cm |
| cv300_gtnorm    | 23.41 / 0.707 / 0.342 | 0.60 | 0.61 | 1.81° / 5.22° | 0.9 cm |
| f300_lidarnorm(平面镜) | 28.56 / 0.824 / 0.137 | 0.73 | 0.74 | — | — |
| c480_lidarnorm(走廊凸面镜) | 23.48 / 0.713 / 0.307 | 0.04 | 0.03 | 2.51° / 6.18° | — |
- 结论:同一个房间换凸面镜,反射照样学起来(β≈0.6,反射占比 0.61,对比 c480 只有 0.03),所以 c480 β 起不来不是"曲面本身"的问题,是走廊场景的问题(镜后没被看到/虚像离墙近/场景大)。
- 镜面画质比平面镜低 4 dB:反射图里房间结构对、但模糊,地板和灯面板在反射里有破洞/边缘锯齿(凸面镜把整个房间压缩到小圆里,对远处表面的分辨率要求高;地板、天花板灯本身也重建得糙)。
- lidarnorm 反而略好于 gtnorm(LPIPS 0.297 vs 0.342),两组法线都 <2°,差别在噪声量级。
- 对比图:scratchpad cv300_cmp.jpg(GT | lidarnorm | gtnorm | 两组 REFLECTION)。

### 2026-11-02 cv300 掉分诊断(cv300_gtnorm,只做推理,data/logs/oracle_cv300.sh)
| 推理改动 | 镜面 PSNR / SSIM / LPIPS-a |
|---|---|
| 原模型 | 23.41 / 0.707 / 0.342 |
| 反射光线用 GT 球面法线+交点(oracle no) | 25.64 / 0.780 / 0.292 |
| 只开子像素多采样 ref_ss=4 | 23.46 / 0.711 / 0.350 |
| GT 几何 + ref_ss=4 | 25.65 / 0.780 / 0.304 |
| GT 几何 + β 强制 1 | 22.06 / 0.746 / 0.315 |
- 高斯拟合球面的误差约占 2.2 dB(法线 med 1.8°/p90 5.2°,反射方向误差翻倍);采样走样不是原因;剩下约 3 dB 不是房间重建差:GT 几何 + β=1(纯反射)画面最清楚、墙纹/物体都对,PSNR 低主要是镜缘一圈亮边(腐蚀 10 px 后 19.3→22.7)和清晰纹理的轻微错位;oracle_no 发糊是因为约 40% 镜面自身颜色是在错误几何下烘焙的模糊反射。=> 关键是训练时就用对的镜面几何(重训),不是房间。
- 镜后 floater:cv300 两组各约 1250 个 op>0.3 高斯在背板后约 16 cm(轮缘平面后 med 21 cm;背板在 3–4.5 cm;f300 只有 89)。45 个背面训练视角都看得到,它冒充无纹理的背板(颜色≈背板,深度偏 10–20 cm),光度 loss 不罚;MirrorBack 深度 MAE 10–12 cm(f300 2.5 cm)。来源推测是 COLMAP 虚像点(此处 14399 个,f300 89)。对策:LiDAR depth loss(r300lp 上 MirrorBack 4.9→1.4 cm)+ 初始化删镜后点。

### 2026-11-02 推理检查:反射直接用 LiDAR 拟合法线图(export_fitted_nmap.py / data/logs/nmap_cv300.sh)
| 模型 / 反射法线 | 镜面 PSNR / SSIM / LPIPS-a |
|---|---|
| gtnorm 原模型 | 23.41 / 0.707 / 0.342 |
| gtnorm + GT 球面法线(oracle n,原点仍用渲染深度) | 24.98 / 0.761 / 0.297 |
| gtnorm + GT 法线图(cv300_gtnorm/normals,同脚本自检) | 24.69 / 0.742 / 0.297 |
| gtnorm + LiDAR 拟合法线图 | 23.02 / 0.712 / 0.311 |
| lidarnorm 原模型 | 24.45 / 0.723 / 0.297 |
| lidarnorm + LiDAR 拟合法线图 | 23.76 / 0.726 / 0.279 |
- 脚本无误(GT 法线图能复现 oracle);LiDAR 法线图不如模型自己的法线。原因:误差集中在镜缘,距轴 0–26° 的 72% 像素 med 0.2–0.34°,
  26–30° 的 20% med 1.34°/p90 2.4°,30–34° 的 8% med 2.78°(80% >2°)。外圈 ~10 cm 是系统偏差(RIM_DROP 后靠 h=0.2 m MLS 外推 +
  镜缘 LiDAR 足迹混到镜框),反射在外圈整体错位,比模型的随机误差更伤。之前报的"内部 0.18°"没覆盖到这一圈。
- 下一步:镜缘外推改用全局参数曲面(整片低阶多项式/二次曲面,稳健拟合内部点再外推),或用可学镜面曲面让光度把外圈修正。

### 2026-11-02 A:全局二次曲面拟合(lidar_mirror_normals_curved.py --surface quadric --rim-drop 0.06 -> cv300_lidarnorm_quadric)
- 镜缘 4 cm 内 LiDAR 格点往里塌(0–2 cm -14.6 mm,2–4 cm -3.7 mm,4–6 cm -1.1 mm),所以丢 6 cm。曲面 vs GT:内部 0.13°,5–10 cm 环 0.50°,外 5 cm 0.77°(MLS 0.18/1.21/2.63)。
  按像素:距轴 26–30° med 0.57°(MLS 1.34),30–34° med 0.83°(MLS 2.78)。掩码 precision 99.91% recall 96.9%。
- 推理检查(反射法线换成法线图):gtnorm 模型 MLS 23.02 -> quadric 23.88(GT 法线图 24.69;GT 法线只用 LiDAR 掩码 24.66,掩码不是差距来源);
  lidarnorm 模型 23.76 -> 24.61(原模型 24.45)。外圈 28% 像素剩 0.6–0.8° 仍掉约 0.8 dB:凸面镜反射对法线非常敏感。
- MeshLab 对比文件:data/result/envgs/mirror_b/mirror_fit_cv300_20261102/(gt_mirror_surface.ply、gt_mirror_assembly.ply、mls/、quadric_rd03|06|10/)。
- 下一步:可学镜面曲面(从 quadric 初始化,LiDAR 锚定内部、光度修外圈),这个 0.8 dB 就是要它拿回来的量。
- 直接与拟合二次曲面求交(export_fitted_quadric.py / data/logs/quadric_cv300.sh,任意视角可用,viewer: gui_fitted_quadric.sh):
  lidarnorm 模型 quadric n 24.97 / 0.757 / 0.266,quadric no(法线+交点)25.79 / 0.772 / 0.260;同模型 GT 球面 no 27.03 / 0.803 / 0.255。
  gtnorm 模型 quadric no 24.63(GT 球面 no 25.64)。法线图版(nmap)只在数据集相机有效,viewer 不能用。
  注意:extent 圆盘半径 0.675 m 大于镜缘 0.541 m(cells 含镜缘混合点),镜框像素也会被替换(β≈0,影响小),待用 LiDAR 掩码边界收紧。
- 命名约定(2026-11-02):`oracle` 只用于用了 GT 的推理对照(export_oracle_sphere.py / gui_oracle_sphere.sh / oracle_*.sh,ORACLE_* 环境变量);
  只用 LiDAR 拟合、不用 GT 的叫 `fitted`:export_fitted_nmap.py / export_fitted_quadric.py / gui_fitted_nmap.sh / gui_fitted_quadric.sh,
  环境变量 FIT_NMAP_ROOT / FIT_QUADRIC / FIT_MODE / FIT_TOL / FIT_TASK / FIT_DEBUG(原 export_oracle_nmap.py / export_oracle_quadric.py 改名而来);
  评估脚本 data/logs/nmap_cv300.sh、quadric_cv300.sh,结果目录 envgs_<run>_nmap_<root>/、envgs_<run>_quadric_<mode>/。

### 2026-11-02 敏感性实验:拟合曲面的标定误差(data/logs/sens_cv300.sh,cv300_lidarnorm,quadric no,只推理)
| 扰动 | 镜面 PSNR / SSIM / LPIPS-a |
|---|---|
| 无扰动 | 25.79 / 0.772 / 0.260 |
| 绕水平轴(俯仰)转 0.2° / 0.5° / 1.0° | 25.51 / 24.29 / 22.30 |
| 绕竖直轴(偏航)转 0.5° | 25.05 |
| 沿法线平移 5 / 10 / 20 mm | 25.63 / 24.74 / 22.90 |
| 沿镜面(水平)平移 10 mm | 24.18 |
| (参考)原模型自己的几何 | 24.45 / 0.723 / 0.297 |
- PSNR 上的保本线约 0.3–0.4° 转动、5–8 mm 平移;LPIPS 更宽容(0.5° 时 0.264 仍好于原模型 0.297)。凸面镜上横向平移也改变法线(10 mm / R 1 m ≈ 0.57°)。
- 真机标定残差常在 0.1–0.3° / 5–20 mm,处在保本线附近 -> 可学镜面曲面里曲面位姿必须可学(6 个参数),LiDAR 只锚形状。

### 2026-11-02 可学镜面曲面(learnable mirror surface,msurf;原"1b")第一批实验,16:15 开训
- 代码:easyvolcap/models/samplers/mirror_surface.py(QuadricSurface / SplineSurface,基类管范围/深度检查/锚定/日志),envgs_sampler.py 的
  mirror_surface_cfg(只替换反射法线,位置仍用高斯深度),mirror_region_supervisor.py 加曲面 loss 和日志。详见 LIDAR_MIRROR_NORMAL.md。
- 拟合导出:data/result/envgs/mirror_b/mirror_surface_cv300_20261102/{quadric_rd06,mls}/patch0.npz。
- 实验(data/logs/launch_cv300_msurf.sh,GPU 0/1/3,训完自动 eval_cv300.sh):
  1) envgs_cv300_msurf_quadric_fixed  2) envgs_cv300_msurf_quadric_learn  3) envgs_cv300_msurf_spline_learn(B 样条,从局部 MLS 初始化)。
- 注意:实验 3 的初始化是局部 MLS(用户原话"优化拟合的局部曲面"),和 1/2 的初始化不同;若要只比较表达能力,补一组 spline 从二次曲面初始化。
- 合成数据没有标定误差,实验 2 预计和 1 接近;二次曲面 9 个自由度被内部 LiDAR 锚住,外圈改不动,它主要用来修位姿。
- **结果(11-03 06:37 全部完成)**:镜面 PSNR / SSIM / LPIPS-a(mirror_perc_b):基线 lidarnorm 24.45 / 0.723 / 0.297;
  1) fixed 30.73 / 0.879 / 0.068;2) quadric_learn **36.76 / 0.968 / 0.033**;3) spline_learn 34.71 / 0.952 / 0.042(平面镜 f300 默认 28.56)。
  曲面法线对 GT(中位 / 镜缘 5 cm):fixed 0.18° / 0.77°;quadric_learn 0.013° / 0.012°(约 15k 步收敛到 GT 球面,锚定残差 1.43→1.46 mm);
  spline 0.022° / 0.045°(镜缘先变差到 9k 步 3.9°,20k 后才恢复,p95 仍 3.1°)。之前"2 和 1 接近"的预计错了。
  注:msurf_gt_* 日志用 float32 acos,小于约 0.03° 显示为 0;精确值是从 ckpt 用 double 算的。
- 待验证(用户提出):镜缘问题可能是镜框太近被光追跳过。tracer 第二跳起最小深度 START_OFFSET=0.06 m(optix_tracer/config.h:40)。
  验证:1) Blender 统计 GT 反射光线 6 cm 内打到镜框/背板的像素分布;2) 单独编译小 offset(如 0.005)的 tracer 用现有 ckpt 渲染对比(注意镜面自相交);3) 确认后小 offset 重训 spline。
- 其他待做:真实场景要从标定误差出发测曲面位姿能否恢复(敏感性 0.3° / 5 mm 即持平基线);LiDAR 深度 + 去初始化点处理背板后 floater;c480 用 quadric 重拟合并上 msurf。


- **镜框验证(11-03)**:
  1) 几何(mirror_lidar/tools/frame_hit_stats.py,Blender GT,val 27 帧,结果 data/result/envgs/mirror_b/frame_check_cv300_20261103/):GT 反射光线 6 cm 内先打到镜框的比例,
     按离镜缘距离:1-2 cm 96%(中位 0.6 cm)、2-3 cm 50%(1.5 cm)、3-5 cm 18%(3.0 cm)、5-10 cm 1%、>10 cm 0;镜框压住最外 1 cm,那里没有可见镜面。
     跳过后看到的多半还是镜框更远处(1-2 cm 带 51%,3-10 cm 带约 100%),1-3 cm 带有 30% 左右变成墙。
  2) 镜面 PSNR 按离镜缘距离分带(mirror_lidar/tools/mirror_rim_band.py;mirror_perc 的镜面 PSNR 腐蚀 4 px,基本不含最外一两厘米):
     quadric_learn 1-2 / 2-3 / 3-5 / 5-10 / >10 cm = 34.9 / 30.3 / 29.4 / 35.0 / 40.3;fixed 33.3 / 26.6 / 24.9 / 28.2 / 34.1;spline 34.6 / 28.8 / 27.6 / 31.2 / 38.1。
     所有组都是 2-5 cm 带最差,正好是"镜框在 6 cm 内、但不是贴着"的那一圈。
  3) 推理时换小 offset(build_variants/dst_off{005,020}:tracer 副本,只改 ptx 常数,已装版本不动;data/logs/eval_cv300_off.sh,结果 envgs_<run>_off{005,020}):
     整体都变差,quadric_learn 镜面 PSNR 36.8 → 29.9(2 cm)/ 19.9(5 mm),镜子内部 >10 cm 带也掉 40.3 → 36.9 / 20.8,所以推理对照结论不干净。
     原因:模型是在 6 cm offset 下训的,镜面上方 0.5-2 cm 有一层高斯(quadric_learn 5480 个,2455 个不透明度 > 0.1),相机看不出来、反射原来又跳过,offset 改小就被斜向反射光线打到;5 mm 时还会打到相邻的镜面面元。
  结论:几何上镜框确实落在被跳过的范围,且最差的带正好对得上;但要证明是原因,得用小 offset 重训(推理替换不行)。

### 2026-11-03 B 样条版两组(用户:都用可学 B 样条,更通用;data/logs/launch_cv300_spline2.sh,12:50 开训,训完自动 eval_cv300.sh)
- GPU0 envgs_cv300_msurf_spline_learn_ldepth005_dz5:+ LiDAR 一回波深度 0.05、死区 5 cm,去背板后 floater(现各组 1000-1600 个不透明高斯,镜缘平面后约 20 cm)。
  cv300_lidar_depth 镜面像素:91% 是表面回波(误差 < 5 cm),5.9% 鬼影(> 5 cm 偏远),同 c480。
- GPU1 envgs_cv300_msurf_spline_learn_off020:反射 START_OFFSET 0.02 m。offset 是 ptx 常数不是配置项:训练/评估/viewer 都要
  PYTHONPATH=build_variants/dst_off020/pkg(启动日志第一行打印实际加载的 tracer 路径)。
- 看:背板后 floater 数、背板深度误差;镜缘分带 PSNR(mirror_rim_band.py)2-5 cm 带;镜面上方 0.5-2 cm 那层高斯是否消失;B 样条镜缘前期是否还变差。
- 待做:LiDAR 空间雕刻(初始点 + 训练中定期)去 floater,未做。
- 结果(11-04,60k;镜面 PSNR / SSIM / LPIPS-a 来自 mirror_perc,背板 = per_obj MirrorBack):
  | 组 | 镜面 | 不透明 PSNR / 深度 MAE | 背板 PSNR / 深度 MAE | 背板后 floater(op>0.1) | 镜缘 1-2 / 2-3 / 3-5 / 5-10 / >10 cm |
  | spline_learn | 34.71 / 0.952 / 0.042 | 40.00 / 5.1 cm | 32.9 / 11.6 cm | 1638 | 34.6 / 28.8 / 27.6 / 31.2 / 38.1 |
  | + ldepth005_dz5 | 34.63 / 0.952 / 0.042 | 40.18 / 1.4 cm | 44.8 / 1.2 cm | 109 | 35.1 / 29.7 / 28.3 / 31.1 / 38.1 |
  | off020 | 34.38 / 0.952 / 0.041 | 39.90 / 5.3 cm | 32.5 / 12.0 cm | 1503 | 32.9 / 27.8 / 27.6 / 30.1 / 38.3 |
  - LiDAR 深度:floater 基本去掉(1638 → 109),背板和整体不透明深度都对了,镜面画质不变。
  - 11-04 用户在 viewer 里看过:效果不错,cv300 场景整体效果很好(当前 cv300 最佳设置)。
  - offset 2 cm:镜缘没变好。frame_check:2-3 cm 带 44% 的反射光线 2 cm 内就打到镜框(仍被跳过),3-5 cm 带 18% 在 2-6 cm(现在能看到),
    但 3-5 cm 带 PSNR 没变(27.6),所以镜缘差主要不是镜框被跳过。镜面上方 0.5-2 cm 那层高斯也还在(三组 8-11k 个,op>0.1 约 5-8k),不是 6 cm offset 造成的。
  - 三组 B 样条镜缘前期都变差(rim5 在约 10k 峰值 3.7-4.2°,20-25k 后回到 < 0.1°),和 offset / 深度无关;最终法线 med 0.02°、rim5 0.03-0.04°。

### 2026-11-03 c480 第 1 步:可学 B 样条镜面(envgs_c480_msurf_spline_learn,GPU3,14:31 开训,data/logs/launch_c480_msurf.sh,训完 eval_c480.sh)
- c480 用全局二次曲面重拟合(--surface quadric --rim-drop 0.06,data/result/envgs/corridor/mirror_surface_c480_20261103/quadric_rd06,日志同目录):
  对 GT 球面内部 0.16° / p95 0.70°,外圈 5 cm 0.71°(原 MLS 2.17°),5-10 cm 0.50°,位置 p95 1.5 mm;
  新数据集 c480_lidarnorm_quadric(掩码精度 100%,召回 86.5%;envs 从 c480_lidarnorm 复制,和基线同一份环境点)。
- 配置 = c480 基线 envgs_c480_lidarnorm_shared_bfilt_mgsn05 + mirror_surface_cfg(spline,2 cm,从二次曲面初始化 = 初始和二次曲面完全一样,
  lr / 锚定 / 正则同 cv300 实验 3),不加 β 损失、不加深度:只看从一开始用对的反射法线 β 能不能起来(基线 / gtnorm 全程 0.03-0.08)。
- 镜框统计(frame_hit_stats.py,ConvexMirror,data/result/envgs/corridor/frame_check_c480_20261103):离镜缘 0-1 / 1-2 / 2-3 / 3-5 cm 的反射光线
  6 cm 内打到镜框 100% / 85% / 33% / 8%,跳过后多数看到墙 / 天花板(颜色变化比 cv300 大);占镜面像素 4.6%,只影响镜缘,不是 β 起不来的原因。
- 结果(11-04 08:04,60k):mirror_beta 2.5k 0.20 → 5k 0.52 → 之后约 0.47(基线 0.04),反射占比 0.48(基线 0.03)。
  镜面 PSNR / SSIM / LPIPS-a 25.93 / 0.815 / 0.143(基线 lidarnorm 23.48 / 0.713 / 0.307,gtnorm 23.59 / 0.721 / 0.289);
  不透明 33.90(基线 33.26),全图 33.43 / 0.907 / 0.113(基线 32.56 / 0.896 / 0.131)。LPIPS 减半,但离 cv300(34-36)还差很远。
  镜面法线 msurf_gt_med 0.20°、rim5 0.68°,学习没有进一步改善(二次曲面初值已接近)。mirror_normal_sphere 的高斯法线变差(med 3.7°)无关:msurf 不用高斯法线反射。
- 11-04 用户在 viewer 里看:镜子中间反射清晰,左右两边很糊,有 floater,镜框重建不好。诊断(mirror_lidar/tools/mirror_lr_bins.py,data/result/envgs/corridor/mirror_lr_c480_20261104):
  - 镜面 PSNR 按水平位置 u(-0.6..+0.6 m,每 0.2 m):msurf 23.6 / 26.3 / 27.7 / 29.3 / 27.7 / 25.2,基线 23.7 / 23.0 / 23.6 / 22.8 / 22.7 / 24.8 —— 中间 +6 dB,最外两带跟基线一样。
    按反射距离 0-1 / 1-2 / 2-4 / 4-8 / >8 m:29.7 / 26.7 / 26.3 / 23.6 / 22.2(远处更差)。
  - 曲面覆盖:msurf 的 extent(LiDAR 1 cm 占据格)缺了镜子上部约 4 行(v > 0.4 只覆盖 11%;LiDAR 在 1.5 m 高、镜子上缘掠射没回波),
    那里回退到高斯法线(med 3.7°)。
  - 法线误差两边大,且学习后变差:|u|>0.4 初值(二次曲面)0.3-1.1°,学完 0.5-1.5°(p90 到 3°);中间 0.1° 不变。cv300 外圈前期变差后能恢复,c480 没恢复(rim5 12.5k 0.51° → 60k 0.68°)。
    1° 法线误差 = 2° 反射方向误差,4 m 外就是 14 cm,对得上"两边糊"。
- 11-04 走廊 vs 房间(同一面 R 1 m 凸面镜)对比(GT persp,val 视角):
  | | 有镜子的训练视角 | 每视角镜面像素(全分辨率)中位 | 相机到镜子中位 | 反射多出的 z 深度 中位 / >4 m |
  | cv300 | 186 / 300 | 925k | 1.6 m | 2.0 m / 0% |
  | c480 | 250 / 480 | 89k(约 1/10) | 4.0 m | 1.4 m / 11% |
  镜子在走廊里小了 10 倍:每个像素在镜面上约 4 mm(cv300 约 1.6 mm),对应反射方向约 0.46°(cv300 0.18°)。单条反射光线采样一个像素 → 混叠/糊,
  训练信号也少 10 倍(β、曲面、反射到的环境都靠它)。判断:主因是镜子小 + 远,其次 LiDAR 法线(远、掠射 → 上部没覆盖、两边 0.3-1.1°)、远处反射内容。
  区分实验:c480 GT 球面法线固定(oracle)+ 深度;若 oracle 也只有约 26 dB,瓶颈不在法线 → 做 ref_ss 多采样。
- 11-04 用户补充:房间绕镜子走了一圈、视角充分,走廊镜子挂墙角基本只在镜前拍。核对(训练视角相机相对镜顶点的方向):
  | | 偏离镜轴 中位 / p90 / 最大 | 绕轴方位(30° 一格,12 格) | 占用 (偏轴 10° × 方位 30°) 格 | 镜顶点处反射方向两两夹角均值 |
  | cv300 | 47° / 68° / 103° | 12 格都有,最少 2 | 50 | 60° |
  | c480 | 39° / 43° / 59° | 3 格空,73% 视角集中在沿走廊水平两向两格,镜子上方方向 0 | 33 | 40° |
  视角少 → 同一镜面点的颜色随视角变化少,反射和镜面本身颜色更难分开(基线 β 塌、反射烤进颜色),曲面法线也只在少数方向上被约束。和"镜子小 10 倍"叠加。

### 2026-11-04 用户三件事
1) 平面镜 f300 + 可学 B 样条:只跑 + LiDAR 深度(envgs_f300_msurf_spline_learn_ldepth005_dz5,GPU3,10:25 开训;不加 LiDAR 那组用户叫停,已杀,无结果)。
   f300 LiDAR 二次曲面拟合(mirror_surface_f300_20261104/quadric_rd06):对 GT 内部 0.034° / 外圈 0.051°,覆盖 100% GT 镜面像素。f300 不跑上限。
2) c480 法线上限(ORACLE,envgs_c480_msurf_oracle_sphere,GPU1):GT 球面固定、整个镜面圆盘(make_oracle_sphere_patch.py),其余同 c480 第 1 步(无深度)。
3) 新走廊场景 ~/SMBC/blender/project/proj_indoor_corridor_convex2(见仓库 SCENES.md),数据集 dataset_c2_480 在 GPU0 渲染(10:58 开始)。
   data/logs/launch_msurf_1104.sh / .log
