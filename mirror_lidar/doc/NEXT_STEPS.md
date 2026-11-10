# 现状与待做

更新 2026-11-10。结果见 `RESULTS.md`。10-22 到 11-05 的完整实验日志在 `archive/EXPERIMENT_LOG_20261022-20261105.md`。

## 现状

- **房间场景已定稿**（11-05，用户确认）：可学 B 样条显式镜面 + LiDAR 第一回波深度。f300 平面镜镜面 PSNR 38.64（基线 28.56），cv300 凸面镜 34.63（基线 24.45）。
- **走廊 c2_480 已定稿**（11-10，用户确认，RESULTS §5）：
  - 定稿组 `envgs_c2_480_msurf_spline_learn_ldepth005_dz5_scons_fitted_voteb03_off020`，不用 GT：
    msurf B 样条 + LiDAR 深度 + 高斯与曲面一致约束，曲面范围由 LiDAR + SAM 掩码多视角投票得到，法线先验和掩码从投票曲面生成（边界带 0.3），反射 offset 2 cm。
  - 镜面 PSNR 28.88、LPIPS-a 0.125（基线 21.78 / 0.491；全给 GT 的 oracleall_off020 29.99 / 0.100）。几何接近 oracle。
  - viewer：`tools/gui.sh fitted_voteb03_off020`。
- **下一个场景**：走廊加大号金属圆柱（blender 仓库 `proj_indoor_corridor_pillar`，R 0.9 m 钢柱，粗糙度 0.05，LiDAR 反射率 0.6、`diffuse_scatter` 0.02），
  400 帧数据集 `dataset_p400` 11-10 在渲染。

## 待做

### 走廊 c2_480（定稿后留下的问题，不急）
- fitted 比 oracle 镜面低约 1 dB，集中在 u −0.4..0 两段；掩码覆盖在这两段正常，可能是单次训练波动。要确认可以换种子重跑 fitted。
- 最右段（u < −0.4）所有组都只有 25–27 dB，输入全给 GT 也一样：视角少、掠射角大。
- vote_b03 掩码离边缘较远的漏标约 1.1%（多在最右段），原因没查（可能是投影时 uncertain / occluded 的标记）。

### 柱子场景（下一步）
- 数据渲完后：检查 LiDAR 回波（柱面应同时有表面回波和鬼影）、准备 EnvGS 数据（COLMAP、StableNormal、LiDAR 深度）。
- 按 c2_480 定稿流程跑：LiDAR 曲面拟合 → SAM 掩码 → 投票范围 → fitted 组；同时跑 oracle 对照。
  柱面是圆柱（只在一个方向弯），注意二次曲面 / B 样条拟合和投票范围对竖长镜面的处理。

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
- 真机标定参数见记忆 `reference_th_s02_real_calibration`：真机镜面约 90% 区域有表面回波 + 鬼影两个回波，仿真原来只有约 0.2%；之后的场景（scatter002 版本起）镜面 LiDAR 材质设 `diffuse_scatter` 0.02，有表面回波。

### 搁置的（试过没收益，或者被更好的方法盖过）
- LiDAR 点云初始化（linit）：收益被 LiDAR 深度监督盖过去。用 LiDAR 法线初始化面元朝向还没试。
- β 软 reset（bsoft）、LiDAR 深度权重衰减（ldecay）、镜面区分法线权重（nsplit）、前期压镜面 RGB（mrgb）、逐像素 β 损失（mbeta，崩）、
  全图或镜面内 2DGS distortion、删镜前面元（prune）。

### 杂项
- 都只本地提交，没 push。`build_variants/`（START_OFFSET 编译变体）和子模块 diff-surfel-tracing 的 validationMode 改动没提交。
- 掩码按 u 分段统计（11-10）用的是临时脚本，没进仓库；要复现可以在 `compare_mirror_masks.py` 里加分段。
