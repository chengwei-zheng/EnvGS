# 现状与待做

更新 2026-11-07。结果见 `RESULTS.md`。10-22 到 11-05 的完整实验日志在 `archive/EXPERIMENT_LOG_20261022-20261105.md`。

## 现状

- **房间场景已定稿**（11-05，用户确认）：可学 B 样条显式镜面 + LiDAR 第一回波深度。f300 平面镜镜面 PSNR 38.64（基线 28.56），cv300 凸面镜 34.63（基线 24.45）。
- **走廊 c2_480 是当前重点**：
  - 最好的组是 `envgs_c2_480_msurf_spline_learn_ldepth005_dz5_scons`（A：+ 高斯与曲面一致约束），镜面 28.71（基线 21.78）。
  - 镜子中部约 32–33 dB，右侧（viewer 图像右边）最外一段只有 25.3，原因是 LiDAR 范围和掩码覆盖不到。

## 在跑（11-07）

| GPU | run | 问题 | 预计 |
|---|---|---|---|
| 0 | `envgs_c2_480_msurf_spline_learn_ldepth005_dz5_scons_dil4_oraclemask` | ORACLE：dil4 + GT 镜面掩码。和 dil4 的差值就是掩码在右侧漏掉 26% 的代价 | 11-07 18:11 起；11-08 10:20 到 epoch 69/120，预计 11-08 夜里训完 |
| 3 | `envgs_c2_480_msurf_spline_learn_ldepth005_dz5_scons_dil4_oraclenormmask` | ORACLE：dil4 + GT 镜面法线先验 + GT 掩码（`c2_480_gtnorm`，`make_gtnorm_sphere.py`）。oraclemask 只换了掩码，法线先验仍是 LiDAR 范围生成的：漏掉的镜面像素还是 StableNormal（虚像法线，对 GT 中位 76°）、边缘带置零跳过。和 oraclemask 的差值就是法线先验的代价 | 11-08 10:54 起，约 33 h |

**dil4 已训完（11-08 00:17）**：镜面 PSNR 28.71 → 29.00、LPIPS-a 0.129 → 0.126，全图 34.26 → 33.99。
最右段只 25.3 → 25.8，没补上；最左段 30.6 → 29.1 反而变差（预处理检查视频里看到外扩后左侧范围压到镜框上）；右侧深度偏差 −0.8 → −1.3 cm。
单靠外扩范围不够，等 oracle 掩码组看掩码的影响。

训完自动跑：`eval_c2_480.sh`、`mirror_lr_bins.py`、`mirror_lr_depth.py`；oracle 组还会跑 `mirror_lr_depth_extent.py`（外扩 4 格）。
结果写到 `data/result/envgs/corridor/mirror_lr_c2_480_scons_20261106/`。

注意：外扩 4 cm 后，右缘仍有 ≤ 1.5 cm 在范围外。这一条即使给了 GT 掩码也没有曲面命中，不受一致约束。

## 待做

### 走廊 c2_480（近期）
1. **对比 A / dil4 / dil4_oraclemask**，看这几项：
   - 镜面指标；
   - 按 u 分段，特别是最右一段；
   - 范围内外的深度偏差；
   - 镜框和镜缘的质量：外扩后镜面面元会不会伸出镜缘、范围外扩到镜框上的那圈反射法线有没有影响。
2. 按结果决定：
   - **掩码差距大**：用外扩后的范围重新生成 LiDAR 掩码，或按边缘自适应外扩（看得少的一侧多扩）。
   - **范围差距大**：改进范围估计。顶部缺约 22 cm，可以用曲面外推加镜框边缘检测。
3. 给最好的组写 viewer 脚本（参照 `tools/gui_c2_480_msurf_ldepth.sh`）。

### 方法（中期）
- **曲面位姿可学**：把曲面拆成"位姿 6 + 形状"，从人为加的标定误差出发，测能不能恢复。
  cv300 推理敏感性：0.3° / 5 mm 就和 2DGS 自己的几何持平，真机标定残差在这个量级。
- **训练时反射多采样**（`ref_ss`）：c480 推理（GT 几何、β 0.9）：K=16 时镜面 PSNR 23.36 → 24.46、LPIPS 0.305 → 0.172。走廊镜子在画面里小，混叠更严重。
- **镜缘**：
  - B 样条镜缘前期变差（约 10k 步出现 4° 的峰），原因没查清；
  - 镜面上方 0.5–2 cm 那层高斯从哪来；
  - START_OFFSET 2 cm 重训没有改善镜缘。
- **没有表面回波时的退路**：用镜框定平面，或者用鬼影翻折。

### 往真机走
- 真机条件退化测试：
  - 只留两个峰；
  - 0.5 m 内的回波合并；
  - 外参误差；
  - LiDAR 与相机时间偏移（最近帧 + 位姿插值）；
  - 干净镜面（表面回波很少）。
- 真机标定参数见记忆 `reference_th_s02_real_calibration`：真机镜面约 90% 区域有表面回波 + 鬼影两个回波，仿真只有约 0.2%。

### 搁置的（试过没收益，或者被更好的方法盖过）
- LiDAR 点云初始化（linit）：收益被 LiDAR 深度监督盖过去。用 LiDAR 法线初始化面元朝向还没试。
- β 软 reset（bsoft）、LiDAR 深度权重衰减（ldecay）、镜面区分法线权重（nsplit）、前期压镜面 RGB（mrgb）、逐像素 β 损失（mbeta，崩）、
  全图或镜面内 2DGS distortion、删镜前面元（prune）。

### 杂项
- 11-08 已本地提交（3fac242，未 push）。`build_variants/`（START_OFFSET 编译变体）和子模块 diff-surfel-tracing 的 validationMode 改动没提交。
