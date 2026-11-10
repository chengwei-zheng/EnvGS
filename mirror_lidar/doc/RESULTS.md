# 已生成的数据和重要结果

更新 2026-11-10（走廊 c2_480 定稿）。按时间记录的实验过程在 `archive/EXPERIMENT_LOG_20261022-20261105.md`（10-22 到 11-05），
更早的在 `mirror_lidar/experiments/*.json` 和 `archive/HANDOFF_EnvGS_20261014.md`；11-06 以后的过程写在本文件和 `NEXT_STEPS.md` 里。

## 1. 场景和数据

### 场景一览
Blender 工程都在 `~/SMBC/blender/project/`，场景说明见 blender 仓库 `SCENES.md`。
EnvGS 场景名就是 `data/datasets/<组>/<名>` 的路径。

| EnvGS 场景名 | Blender 数据集 | 镜子 | 用途 / 状态 |
|---|---|---|---|
| `bathroom/walk400v2`、`walk400_2mirrors` | proj_indoor_bathroom / dataset_walk400{,_2mirrors} | 圆镜 + 墙镜 + 淋浴玻璃 | 最早的检查（10-14）：镜子全变虚像房间。不再用 |
| `mirror_a/a220` | proj_indoor_mirror_a / dataset_a220 | 墙镜，背面永远看不见 | 基线变虚像房间（10-15）。不再用 |
| `mirror_b/b300` | proj_indoor_mirror_b / dataset_b300 | 立镜，背面可见 | 共用高斯首次验证（10-17）。不再用 |
| `mirror_b/m300` | proj_indoor_mirror_b_mirror30 | B 只转镜子 30° | gtnorm 对照（10-20）。不再用 |
| `mirror_b/r300` | proj_indoor_mirror_b_rot30 / dataset_r300 | B 转 30° + 家具挡镜 | **灯有 bug**（面光源相机看不见、镜中看得见），只作参考 |
| `mirror_b/r300lp` | 同上 / dataset_r300_lamppanel | 同上，灯换成发光面板 | 10-22 到 10-28 的主要实验场景 |
| `mirror_b/f300` | proj_indoor_mirror_b_framed / dataset_f300 | 3 cm 厚平面镜 + 木框 | **房间平面镜定稿场景** |
| `mirror_b/cv300` | proj_indoor_mirror_b_convex / dataset_cv300 | f300 换成 R 1 m 凸面镜 | **房间凸面镜定稿场景** |
| `corridor/c480` | proj_indoor_corridor_convex / dataset_c480 | 走廊墙角凸面镜 R 1 m | 走廊 v1，视角覆盖差 |
| `corridor/c2_480` | proj_indoor_corridor_convex2 / dataset_c2_480 | 走廊凸面镜 R 1.125 m，宽 1.35 m | **走廊定稿场景**（11-04 到 11-10，用户 11-10 确认） |

GT 球面：
- cv300、c480 在各自工程目录的 `mirror_sphere.json`（c480 中心 (−0.02669, 2.42669, 2.25706)，R 1.0）；
- c2_480 在 `proj_indoor_corridor_convex2/mirror_sphere.json`：中心 (−0.04195, 2.44195, 2.23294)，R 1.125，半张角 36.87°。

### 数据目录（`~/SMBC/EnvGS/data/`，仓库里的 `data` 是指向它的软链）

| 目录 | 内容 | 大小 |
|---|---|---|
| `datasets/original/<组>/<场景>/` | COLMAP 工程（GT 位姿三角化），`<场景>.colmap.log` | 13 G |
| `datasets/<组>/<场景>/` | EasyVolcap 格式 + StableNormal | 共 25 G |
| `datasets/<组>/<场景>_lidar_depth/` | LiDAR 第一回波深度（每相机一个 npz） | |
| `datasets/<组>/<场景>_lidarnorm[_quadric]/` | 训练用：LiDAR 镜面法线先验 + `mirror_masks/` + `envs/`；其余文件是软链 | |
| `datasets/<组>/<场景>_lidarnorm*_debug/` | 拟合预览图、`eval.json` | |
| `datasets/<组>/<场景>_gtnorm/`、`_nmask/` | oracle / 消融法线先验（GT 镜面法线、镜面置黑） | |
| `datasets/corridor/c2_480_oracle_mirror_masks/` | GT 镜面掩码（oracle，11-07） | |
| `datasets/corridor/c2_480_lidarnorm_vote{,_b03}/` | **不用 GT 的定稿输入**（11-09）：投票范围曲面（`mirror_extent_vote.py`）生成的法线先验和掩码；`_b03` 边界带 0.3 倍（`--band 0.3`），fitted 定稿组用它 | |
| `datasets/mirror_b/r300lp_lidarnorm_linit/` | LiDAR 点云初始化 | |
| `datasets/mirror_b/r300lp_lidar_depth_{mplane,nomirror}/` | 镜面 LiDAR 样本换成拟合平面深度 / 去掉 | |
| `datasets/envgs/`、`refnerf/`、`mmw421_samsung/` | 官方数据（audi 复现过）/ 没用 | |
| `trained_model/envgs/<组>/envgs_<run>/` | checkpoint（`latest.pt`） | 70 G |
| `record/envgs/<组>/envgs_<run>/` | tensorboard + 解析后的配置 | 3.5 G |
| `result/envgs/<组>/envgs_<run>/` | 训练结束时的测试渲染、`metrics.json`；评估后多出 `rawdepth/`、`eval_npz/`、`eval_gt/`（`summary_val.json`） | 87 G |
| `result/envgs/<组>/envgs_<run>_<tag>/` | 推理对照（oracle / fitted / off / prune 等），原模型不动 | |
| `vis/` | 早期 r300 / r300lp 的对比图、点云可视化 | 111 M |

`EnvGS/data/logs/`（仓库内，gitignore）：
- `prep_<场景>.sh`：数据准备；
- `launch_*.sh` / `.log`：训练 + 自动评估的启动脚本；
- `eval_<场景>.sh`：评估；
- `cmd_envgs_<组>_<run>.txt`：精确的训练命令；
- `envgs_<组>_<run>_train.log`；
- 各分析脚本的 `.log`。

### 分析结果目录（`data/result/envgs/<组>/` 下，不以 `envgs_` 开头的）

| 目录 | 内容 |
|---|---|
| `mirror_b/mirror_fit_cv300_20261102/` | cv300 拟合曲面对比：GT 镜面网格、MLS、二次曲面 rd03/06/10，MeshLab 可直接打开 |
| `mirror_b/mirror_surface_cv300_20261102/`、`mirror_surface_f300_20261104/` | msurf 初始化用的 `patch0.npz`（f300 用 `quadric_rd06_fix`） |
| `mirror_b/frame_check_cv300_20261103/` | 反射光线 6 cm 内打到镜框的统计 |
| `mirror_b/beta_check_f300_20261104/` | f300 β 死区诊断（拟合工具 bug） |
| `mirror_b/figs_cv300_20261102/` | cv300 对比图 |
| `corridor/mirror_surface_c480_20261103/`、`mirror_surface_c2_480_20261104/` | 走廊二次曲面拟合（`quadric_rd06/patch0.npz`）、拟合和掩码日志 |
| `corridor/frame_check_c480_20261103/` | c480 镜框统计 |
| `corridor/mirror_oracle_c480_20261104/` | c480 GT 球面 patch（oracle） |
| `corridor/mirror_lr_c480_20261104/`、`mirror_lr_c480_oracle_20261105/` | c480 镜面按水平位置 / 反射距离分段的 PSNR |
| `corridor/mirror_lr_c2_480_20261106/` | c2_480 分段 PSNR（`bins.txt`）、渲染深度（`depth.txt`）、LiDAR 样本（`lidar.txt`） |
| `corridor/mirror_lr_c2_480_scons_20261106/` | 一致约束各组：`bins_*`、`depth_*`、`depth_extent*`（按范围内外分）、`mask_lr.txt`（掩码按位置的召回） |
| `corridor/prep_check_c2_480_20261107/` | 预处理检查视频（`vis_mirror_prep.py`）：曲面初始化投影（原范围 / 外扩 4 cm）、LiDAR 掩码 + GT 轮廓、oracle 掩码 |
| `corridor/spline_normal_c2_480_20261106/`、`_scons_20261107/` | 学到的 B 样条对 GT 球面的法线误差（`summary.txt`、`normal_err_0to5deg.png`、`extent_vs_gt.png`） |
| `corridor/mirror_surface_c2_480_oracle_20261108/` | oracleall 用的 GT 球面 patch（`make_oracle_patch.py`） |
| `corridor/sam_mask_c2_480_20261109/`、`sam_mask_c2_480_maskprompt_20261109/` | SAM 3 镜面掩码（LiDAR 提示），`metrics.txt`、`vis_zoom/`；定稿用 `box_pos_iter` |
| `corridor/mirror_extent_vote_c2_480_20261109/v3_pad25_frac03/` | **多视角投票的镜面范围**：`patch0.npz`（fitted 定稿组的 msurf 曲面）、`report.txt`（对 GT 镜面的格子 P/R/IoU、每 30° 扇区的边缘缺口） |
| `corridor/mask_compare_c2_480_20261109/`、`mask_compare_c2_480_b03_20261109/` | 各掩码对 GT 的比较（`compare_mirror_masks.py`）：`report.txt`、叠加图 |

**11-08 清理磁盘**（清单 `data/logs/cleanup_candidates_20261108.txt`）：
- 删了所有 `rawdepth*` / `rawspec`（`export_raw.py` 的中间产物，已经转成 `eval_npz`；重跑 `eval_*.sh` 会重新生成）。
- 删了修灯 bug 之前所有场景的 result 和 checkpoint：bathroom、mirror_a、mirror_b 的 b300 / m300 / r300（不含 r300lp）。这些实验的结论只在本文档和 archive 里，不能再开 viewer 或重新评估。

## 2. 主要结论（按时间）

1. **原版 EnvGS 学不出镜面**（浴室、场景 A，10-14 / 10-15）：
   - 镜面区域约 80% 的深度落在镜后虚像房间里，反射分支基本不用；
   - "镜后是一个虚像房间"能精确解释每个视角，纯 RGB 没有理由选择"镜面 + 反射"。
2. **共用一套高斯做反射可行**（b300，10-17）：反射光线追主面元、取 DC 颜色。
   - 有正确镜面法线时收敛，镜面 PSNR 26.72 → 28.83，LPIPS 0.094 → 0.065，镜中明显更清晰；
   - 但它更依赖镜面法线：法线先验错或没有时，镜子会变成贴图。
3. **镜面法线先验是关键**（r300，10-19）：
   - 镜子转 30° 后，StableNormal（镜面中位误差 45°）和不加先验的组都变成虚像房间（镜面平面比例 6.5% / 8.1%）；
   - 只有 GT 法线组正确（99.9%），不透明区还 +7 dB。
4. **LiDAR 拟合的镜面法线可以代替 GT 法线**（r300，10-21）：各项和 gtnorm_nw5 持平（PSNR 35.92 vs 35.84）。
   - 加上 LiDAR 第一回波深度后，不透明深度中位误差 6.1 → 0.3 cm，天花板 0.35 m → 3 mm。
5. **bfilt 只追 β 大的像素**（r300lp，10-23）：快 2.3 倍，画质不变，之后默认开启。
6. **镜面鼓包**（r300lp / f300，10-26 到 10-29）：
   - 直接删镜前面元有害；
   - 镜面内 gs_norm 加到 1.0 时镜面面元伸出镜缘、墙面掉 5 dB；全图 distortion 伤不透明区；
   - f300 权重扫描后，用户定 0.5（`_mgsn05`）为房间默认。
   - 训练末段镜面指标的波动约 LPIPS ±0.01、PSNR ±0.8 dB。
7. **曲面镜上法线先验不够**（c480，10-30）：β 起不来（0.03–0.08），反射被烘焙进镜面颜色。
   - 加深度只改善几何；
   - 逐像素 β 损失会长出相机近处的浮游物，训练崩溃。
8. **同一房间换凸面镜照样学得起来**（cv300，11-02）：β 约 0.6。说明 c480 失败在走廊场景，不在曲率。
   - 剩下的误差来自高斯拟合的镜面法线（1.4–1.8°）。
9. **显式镜面 msurf**（cv300，11-03）：反射法线换成拟合曲面，固定不学就 +6.3 dB，可学 +10 dB。
   - 曲面能收敛到 GT 球面（二次曲面 0.013°）。
10. **房间场景定稿**（11-05，用户确认）：msurf B 样条 + LiDAR 深度。见下表。
11. **走廊 c480 上限**（11-05）：GT 球面法线（oracle）也只有 27.6 dB，瓶颈在视角覆盖（镜子挂墙角、只在镜前拍、镜子在画面里小 10 倍）。于是换成 c2_480。
12. **走廊 c2_480**（11-06 / 11-07）：msurf + LiDAR + 高斯与曲面一致约束，镜面 21.8 → 28.7。见下表。
13. **走廊 c2_480 定稿**（11-10，用户确认）：不用 GT，只靠 LiDAR + SAM 掩码投票得到镜面范围（`fitted_voteb03_off020`），
    几何接近全给 GT 的 oracle（法线 >5° 16% vs 13%），镜面 PSNR 28.88、LPIPS-a 0.125（oracle 29.96 / 0.100，基线 21.78 / 0.491）。见 §5。

## 3. 房间场景定稿（11-05）

| 场景 | 设置 | 全图 PSNR / SSIM / LPIPS | 镜面 SSIM / LPIPS-a / PSNR | 镜面 β | viewer |
|---|---|---|---|---|---|
| f300 平面镜 | 基线 `envgs_f300_lidarnorm_shared_bfilt_mgsn05` | 36.00 / 0.943 / 0.085 | 0.824 / 0.137 / 28.56 | 0.73 | `tools/gui_f300_best.sh` |
| f300 平面镜 | **`envgs_f300_msurf_spline_learn_ldepth005_dz5_fitfix`** | **39.82 / 0.968 / 0.057** | **0.973 / 0.022 / 38.64** | 0.80 | **`tools/gui_f300_msurf_best.sh`** |
| cv300 凸面镜 | 基线 `envgs_cv300_lidarnorm_shared_bfilt_mgsn05` | 33.99 / 0.935 / 0.101 | 0.723 / 0.297 / 24.45 | 0.61 | |
| cv300 凸面镜 | **`envgs_cv300_msurf_spline_learn_ldepth005_dz5`** | **38.95 / 0.967 / 0.059** | **0.952 / 0.042 / 34.63** | 0.79 | **`tools/gui_cv300_msurf_best.sh`** |

- 不透明区不变或略好：f300 39.68 → 40.30，cv300 39.71 → 40.18。
- cv300 上可学二次曲面（`envgs_cv300_msurf_quadric_learn`）镜面 36.76，比 B 样条高 2 dB；固定二次曲面 30.73。
- LiDAR 深度不改镜面画质（cv300 34.71 → 34.63），但镜后 floater 1638 → 109 个，背板深度 MAE 11.6 → 1.2 cm。

## 4. 走廊 c480（10-29 到 11-05）

| 组 | 镜面 SSIM / LPIPS-a / PSNR | 全图 PSNR | 镜面 β |
|---|---|---|---|
| 基线 `envgs_c480_lidarnorm_shared_bfilt_mgsn05` | 0.713 / 0.307 / 23.48 | 32.56 | 0.04 |
| GT 法线先验 `envgs_c480_gtnorm_shared_bfilt_mgsn05` | 0.721 / 0.289 / 23.59 | 32.71 | 0.03–0.08 |
| `envgs_c480_msurf_spline_learn` | 0.815 / 0.143 / 25.93 | 33.43 | 0.47 |
| `envgs_c480_msurf_oracle_sphere`（ORACLE：GT 球面，不学） | 0.841 / 0.125 / 27.62 | 33.43 | 0.50–0.55 |

- 按水平位置 u 分段（−0.6 到 +0.6 m，每 0.2 m 一段）：
  - msurf：23.6 / 26.3 / 27.7 / 29.3 / 27.7 / 25.2；
  - oracle：22.9 / 29.1 / 30.4 / 32.3 / 32.0 / 27.4；
  - 两边差，且最左一段 oracle 也救不回来。
- 和房间 cv300 比：
  - 镜子在画面里约小 10 倍（每视角镜面像素中位 89k vs 925k）；
  - 视角集中在沿走廊的两个方向（偏离镜轴 p90 43° vs 68°）。

## 5. 走廊 c2_480（11-04 到 11-10，已定稿）

评估：`data/logs/eval_c2_480.sh`（60 个验证视角）；分段统计用 `tools/mirror_lr_bins.py`。

| 组（`envgs_c2_480_...`） | 全图 PSNR | 镜面 SSIM / LPIPS-a / PSNR | 镜面 PSNR 按 u 分 6 段（右 → 左*） | 右侧深度偏差** |
|---|---|---|---|---|
| 基线 `lidarnorm_quadric_shared_bfilt_mgsn05` | 32.04 | 0.642 / 0.491 / 21.78 | 23.5 / 22.3 / 20.9 / 20.6 / 21.1 / 21.5 | +0.9 cm |
| `msurf_spline_learn` | 33.32 | 0.801 / 0.192 / 26.17 | 23.6 / 25.4 / 28.8 / 30.2 / 30.6 / 27.0 | −8.5 cm |
| `msurf_spline_learn_ldepth005_dz5` | 34.12 | 0.831 / 0.150 / 27.89 | 24.7 / 27.9 / 32.8 / 33.0 / 32.6 / 29.3 | −4.3 cm |
| **`..._ldepth005_dz5_scons`**（+ 一致约束，A） | **34.26** | **0.849 / 0.129 / 28.71** | 25.3 / 32.3 / 32.7 / 32.6 / 32.4 / 30.6 | −0.8 cm |
| `..._scons_dist10`（A + 镜面 distortion 10，B） | 33.54 | 0.819 / 0.165 / 27.12 | 23.8 / 27.7 / 30.3 / 29.4 / 30.9 / 28.7 | −3.2 cm |
| `..._scons_dil4`（A + 范围外扩 4 cm） | 33.99 | 0.847 / 0.126 / 29.00 | 25.8 / 32.2 / 33.4 / 33.2 / 32.6 / 29.1 | −1.3 cm |
| `..._scons_dil4_oraclemask`（ORACLE：dil4 + GT 镜面掩码） | 33.58 | 0.833 / 0.159 / 28.11 | 23.6 / 27.4 / 30.3 / 29.1 / 31.3 / 30.2 | −1.0 cm |
| `..._scons_dil4_oraclenormmask`（ORACLE：dil4 + GT 法线先验 + GT 掩码） | 34.08 | 0.852 / 0.123 / 29.45 | 25.4 / 32.3 / 32.6 / 33.4 / 32.7 / 29.6 | −1.3 cm |
| `..._scons_oracleall`（ORACLE：GT 法线先验 + GT 掩码 + GT 曲面和范围，反射 offset 6 cm） | 34.21 | 0.862 / 0.111 / 29.95 | 27.7 / 30.5 / 32.3 / 32.9 / 33.5 / 30.5 | −0.4 cm |
| `..._scons_oracleall_off020`（同上，反射 offset 2 cm） | **34.53** | **0.868 / 0.100 / 29.99** | 27.3 / 29.6 / 32.7 / 32.7 / 32.5 / 31.3 | −0.0 cm |
| **`..._scons_fitted_voteb03_off020`（定稿，不用 GT：投票范围曲面 + 其法线先验和掩码（边界带 0.3），offset 2 cm）** | 34.14 | 0.855 / 0.125 / 28.88 | 26.2 / 27.5 / 30.6 / 33.1 / 32.5 / 30.0 | −0.2 cm |

\* u 从 −0.6 到 +0.6 m，第一段 u < −0.4，是 **viewer 图像右侧**。
\** 右侧（u < −0.4，范围内）的渲染深度减曲面交点，负值表示高斯在曲面前方。

- **一致约束（A）**：把右侧比曲面靠前 4–5 cm 的高斯层拉回曲面（−4.3 → −0.8 cm）。
  - 右侧次外段 +4.3 dB，左侧最外段 +1.3 dB；
  - 学到的曲面法线 p90 从 0.32° 降到 0.16°，右侧中位 0.35° → 0.15°；
  - viewer：`tools/gui_c2_480_msurf_ldepth.sh` 打开的是没有一致约束的组；A 组目前没有单独的 viewer 脚本。
- **镜面 distortion（B）反而掉分**：中部从 32–33 掉到 29–30。
- **最右一段仍只有 25.3**，原因是范围 / 掩码覆盖不到：
  - LiDAR 范围在右缘缺 1–5.5 cm，顶部缺约 22 cm；
  - 右侧掩码召回只有 74%（边界带 13%、漏掉 13%），全镜 91%。
  - dil4 和 dil4_oraclemask 两组就是要分开"范围"和"掩码"这两个因素。
- **11-09 oracle 结果**：
  - **oraclemask 反而比 dil4 差**（镜面 −0.9 dB，LPIPS-a 0.126 → 0.159），中间几段掉 3–4 dB，最右段 25.8 → 23.6。只换掩码时，GT 掩码覆盖到的像素有一部分法线先验还是 StableNormal（虚像法线）或置零带，掩码和法线先验对不上；原因还没确认。
  - **oracleall_off020（输入全给 GT）是目前 c2_480 最好的一组**：全图 34.53，镜面 LPIPS-a 0.100；最右段 27.3（dil4 25.8），最左段 31.3（29.1）；右侧深度偏差 −1.3 → −0.0 cm，> 2 cm 的比例 45% → 16%。
    但右侧次外段 29.6（dil4 32.2）。输入全对时最右段仍比中部低约 5 dB，所以右侧差的主因不在输入（视角少、掠射角大）。曲面范围、法线先验、2 cm offset 三个因素混在一起，没拆开。
- **曲面学得很准**：B 样条对 GT 球面的法线误差中位 0.07°（初值 0.15°）。所以剩下的问题在高斯，不在曲面。
- **11-10 oracle 收尾**：
  - oraclenormmask（GT 法线先验 + GT 掩码）比 dil4 好一点（镜面 29.08 → 29.45），比只换掩码的 oraclemask 好得多：oraclemask 变差是掩码和法线先验对不上造成的。最右段仍 25.4，右侧深度偏差仍 −1.3 cm：右侧靠 2D 输入补不上，要靠曲面范围。
  - offset 2 cm 对 6 cm：镜面 PSNR 一样（29.99 vs 29.95），LPIPS-a 0.100 vs 0.111，法线 >5° 12.6% vs 24.9%，右侧深度偏差 −0.0 vs −0.4 cm。定稿用 2 cm。
  - 两组 oracle 同一分段差到 1 dB（右侧次外段 29.6 vs 30.5、左二段 32.5 vs 33.5），这是单次训练的波动量级。

### c2_480 定稿：不用 GT 的 fitted 组（11-09 / 11-10）

输入的做法（全部只用 LiDAR + RGB）：
1. **SAM 3 镜面掩码**（`sam_mirror_mask.py`）：LiDAR 掩码做提示（box + 正点 + 迭代），和 LiDAR 掩码融合（L+sam）。
2. **多视角投票镜面范围**（`mirror_extent_vote.py`，`v3_pad25_frac03`）：LiDAR 拟合的二次曲面外推 25 cm，每个 1 cm 格子在所有看得到它的训练视角里按 2D 掩码投票（按入射角和像素尺度加权，阈值 0.3）；LiDAR 单回波占多数的格子否决（镜框、墙：多回波比例中位 0.01，镜面边缘 0.76）。
   - 对 GT 镜面（半径 0.669 m）：格子 P 100% / R 97.1% / IoU 97.1%（LiDAR 范围 84.8%、dil4 86.2%），每个扇区边缘缺口 0.6–2.6 cm（LiDAR 范围顶部缺 18.8 cm、右侧 3.9 cm）。
   - 曲面法线对 GT 中位 0.15°，外圈 5 cm 0.84°；径向位置误差中位 −0.2 mm。
3. **法线先验和掩码从投票曲面重新生成**（`lidar_mirror_normals_curved.py --patch` + `lidar_mirror_mask.py --band 0.3`）。

掩码对 GT（`compare_mirror_masks.py`，301 个有镜子的视角；内部 = 255，"+带" 把边界带也算命中）：

| 掩码 | P | R | IoU | mIoU | 右侧 R | 轮廓到 GT 边缘平均 px |
|---|---|---|---|---|---|---|
| LiDAR（旧） | 99.19 | 91.03 | 90.37 | 81.95 | 75.4 | 45.4 |
| L+sam | 94.85 | 97.35 | 92.46 | 91.30 | 92.7 | 19.8 |
| vote（默认带宽） | 99.98 | 93.90 | 93.88 | 89.51 | 85.3 | 35.8 |
| vote +带 | 98.21 | 99.19 | 97.43 | 94.80 | 97.6 | 7.0 |
| **vote_b03** | 99.92 | 96.80 | 96.73 | 95.26 | 91.2 | 9.1 |
| **vote_b03 +带** | 99.64 | 98.29 | 97.94 | 97.38 | 95.0 | 5.2 |

- SAM 融合的掩码有 3–4% 远离镜缘的误检（P 94.85）；投票后基本没有误检。
- vote_b03 在训练视角按 u 分段的内部覆盖：最右段 91.2%（3.8% 在边界带、3.0% 是边缘 10 px 内、2.1% 远离边缘漏掉），其余各段 95.6–98.1%。

**fitted 组结果**（表中最后一行）：
- 几何接近 oracle：镜面法线中位 2.19°、>5° 16.2%（oracleall_off020 2.00° / 12.6%，oracleall 2.31° / 24.9%，dil4 2.41° / 27.6%）；镜面深度 |err| > 2 cm 最右段 20.9%（16.1% / 29.5%，dil4 42%）；渲染镜面 99% 落在投票范围内；训练末 `msurf_cons_frac` 1.0，学到的曲面对 GT 中位 0.07°。
- 画质：全图 34.14，镜面 28.88 / LPIPS-a 0.125，和 dil4 相当，比 oracle 低约 1 dB。差距在 u −0.4..0 两段（−2.1 dB）；这两段的掩码覆盖和左侧一样（97–98%）、深度也不比 oracle 差，所以不是掩码造成的，可能是单次训练波动（两组 oracle 之间也有 1 dB），没再拆。
- 用户 11-10 看过 viewer，认为结果可以，**c2_480 到此定稿**。viewer：`tools/gui.sh fitted_voteb03_off020`。

## 6. viewer 脚本（`mirror_lidar/tools/`，在 GPU2 上开，关窗不保存）

| 脚本 | 打开 |
|---|---|
| **`gui.sh <run>`** | **通用（11-09 起推荐）**：run 写全名或能唯一确定的一段（如 `oracleall_off020`、`oraclenormmask`），自动找组、viewer 配置、反射偏移版本（名字带 `_off020` 就用 2 cm 光追）；`gui.sh -l [关键词]` 列出训练好的 run，`gui.sh -n <run>` 只打印不打开；同时只允许开一个 viewer |
| `gui_f300_msurf_best.sh` / `gui_cv300_msurf_best.sh` | 房间定稿 |
| `gui_f300_best.sh` | f300 基线（mgsn05） |
| `gui_c2_480_msurf_ldepth.sh` | c2_480 B 样条 + LiDAR 深度 |
| `gui_mirror_b.sh <run>` | 通用：mirror_b 组任意 run；加 `GROUP=corridor VIEWER=corridor_c2_viewer` 可以开走廊的 run |
| `gui_fitted_quadric.sh` / `gui_oracle_sphere.sh` / `gui_fitted_nmap.sh` | 推理时替换反射几何的对照 |
| `gui_bathroom.sh` / `gui_mirror_a.sh` | 早期场景 |
