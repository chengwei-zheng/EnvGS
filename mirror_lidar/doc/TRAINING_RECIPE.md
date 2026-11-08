# EnvGS 训练流程：当前设置、渲染步骤、特殊操作与 loss

更新 2026-11-07。2026-10-25 版的旧文档在 `archive/TRAINING_RECIPE_20261105.md`，里面有逐行核对代码的细节和各组 r300 / r300lp 实验表。
数据怎么准备见 `DATA_PREP.md`，显式镜面的原理见 `MIRROR_SURFACE.md`，结果见 `RESULTS.md`。

## 当前主线配置

**方法 = 2DGS 面元（共用一套高斯做反射）+ 可学 B 样条显式镜面（msurf）+ LiDAR 第一回波深度 + 高斯与曲面一致约束。**

| 场景 | 配置（`configs/exps/envgs/...`） | 说明 |
|---|---|---|
| 走廊 c2_480（**当前重点**） | `corridor/envgs_c2_480_msurf_spline_learn_ldepth005_dz5_scons.yaml` | 11-07 最好（镜面 28.71） |
| 房间 f300 平面镜 | `mirror_b/envgs_f300_msurf_spline_learn_ldepth005_dz5_fitfix.yaml` | 11-05 定稿（镜面 38.64），没有一致约束 |
| 房间 cv300 凸面镜 | `mirror_b/envgs_cv300_msurf_spline_learn_ldepth005_dz5.yaml` | 11-05 定稿（镜面 34.63），没有一致约束 |

**配置继承链**：`configs/base.yaml` → `configs/models/envgs.yaml` → 数据集配置（如 `configs/datasets/corridor/c2_480_lidarnorm_quadric.yaml`）→ 实验 yaml。
实验 yaml 开头的注释按日期写了这组在上一组基础上改了什么，从 r300lp 一路累积下来。

**相对 EnvGS 原版的改动一览**（c2_480 主线；后面各节详述）：

| 项 | 设置 | 来源 |
|---|---|---|
| 共用高斯反射 | `shared_reflection: True`（默认）：反射光线追主面元本身、取 DC 颜色，不用环境高斯 | 10-17 |
| 只追 β 大的像素（bfilt） | `specular_filtering_start_iter 4500`、`threshold 0.02`、`pause_iter 1500` | 10-22 |
| 法线先验 | 权重 0.05（原版 0.01），镜面用 LiDAR 拟合法线，`norm_ignore_zero` 跳过边界带 | 10-20 |
| 镜面内深度-法线一致 | `mirror_gs_norm_loss_weight 0.5`，3000 起 | 10-27 / 10-29 |
| LiDAR 第一回波深度 | `lidar_dpt_loss_weight 0.05`、`lidar_dpt_dead_zone 0.05` | 10-20 / 10-26 |
| 显式镜面 msurf | B 样条 2 cm，4500 起可学，lr 2e-6，锚定 0.01，正则 1e-4 | 11-02 |
| 高斯与曲面一致 | `msurf_dpt 1.0`、`msurf_norm 0.5`，3000 起 | 11-06 |
| LPIPS 起始 | `perc_loss_start_iter 45000`（原版 21000，为了加速） | 10-14 |

**代码位置**：
- 采样器：`easyvolcap/models/samplers/envgs_sampler.py`（继承 `gaussian2d_sampler.py`）、`mirror_surface.py`；
- 高斯操作：`easyvolcap/utils/gaussian2d_utils.py`；
- loss：`easyvolcap/models/supervisors/{volumetric_video,envgs,lidar_depth,mirror_region}_supervisor.py`。

---

## 0. 基本设置

| 项 | 值 |
|---|---|
| 总迭代 | `epochs 120 × ep_iter 500 = 60000`，batch 1，每次一张完整图 |
| 分辨率 | `ratio 0.5` → 1296×972 |
| 训练 / 验证视角 | 每 8 张取 1 张做验证。c2_480 是 420 / 60，f300 / cv300 是 262 / 38 |
| 训练时间（3090） | c2_480：三组同时跑约 22–30 h；房间场景 bfilt 单独跑约 9–11 h |
| 场景尺度 | `spatial_scale`（c2_480 4.87、f300 约 2），带 × spatial_scale 的阈值都乘它 |
| 优化器 | Adam eps 1e-15。xyz 1.6e-4 × spatial_scale 指数衰减到 1/100（3 万次衰减完）；DC 2.5e-3、SH rest 1.25e-4、opacity 0.05、scaling 5e-3、rotation 1e-3、β 0.01；msurf_h 2e-6 |

### 初始化
- **主面元**：`sparse/0/points3D.ply`，由 COLMAP 在 GT 位姿下三角化，只用 RGB。
  - DC = 点颜色，SH 高阶为 0；两轴尺度 = 到最近 3 个点的均方根距离；
  - 旋转 = normalize(torch.rand(4))：随机，而且不均匀；
  - opacity 0.1，β 0.001。
- **COLMAP 点里有镜中虚像点**：r300 8.7%，cv300 镜后 1.4 万个。
  - LiDAR 点云初始化（`lidar_init_points.py`）能降到 0.14%；
  - 但它的收益被 LiDAR 深度监督盖过去了，所以没有进主线。
  - LiDAR 深度监督也能去掉镜后 floater（cv300 从 1638 个降到 109 个）。
- **环境高斯**：`envs/points3D.ply`，第一次训练时在 env_bounds 内随机生成。共用版不用它。
- **显式镜面**：从 `patch0.npz` 初始化，初始形状和 LiDAR 二次曲面拟合完全一样。

## 1. 每次迭代的渲染（`EnvGSSampler.forward`）

**每个像素只有一条相机光线和一条反射光线。** 反射是理想镜面方向，没有粗糙度采样，也没有二次反射。
但一条反射光线不只取第一个命中点，而是把沿途面元按 alpha 从前往后混合。

1. **训练期操作**：`update_dif_gaussians`，见 §2。
2. **相机光线**：取像素中心。d 用 **z 深度约定**（相机系 z 分量为 1，|d| = 1/cosθ），所以 o + d·z 就是 z 深度为 z 的点。
3. **主面元栅格化**（2DGS，`diff-surfel-rasterization-wet-ch05`）：
   - 光线和面元平面求交，G = exp(−½|uv|²)，α = min(0.99, opacity·G)，从前往后混合，权重 w_i = α_i·T_i；
   - 输出：
     - `rgb_map`：c_base；
     - `acc_map`；
     - `dpt_map`：期望 z 深度 Σw·z / Σw（`depth_ratio 0`）；
     - `norm_map`：Σw·n，世界系，没有归一化；
     - `surf_norm_map`：深度有限差分法线；
     - `spec_map`：Σw·β；
     - `dist_map`：2DGS distortion。
4. **反射光线**（3000 次起，`get_reflect_rays`）：
   - 起点 `ref_o = o + d·dpt_map`（渲染表面点，不 detach）。
   - 方向 `ref_d = d − 2(d·n)n`，不归一化，和 tracer 的 z 深度约定一致。
   - n 一般是 normalize(norm_map)；**开了 msurf 时，镜面像素换成曲面法线**：相机光线在范围内命中曲面，且 |t − dpt| < 0.15 m。
   - bfilt 筛选期间只取 β > 0.02 的像素，见 §2.5。
5. **OptiX 追踪**（`diff-surfel-tracing`）：
   - 每个面元是一个 ±3σ 的四边形，BVH 每次前向都重建；
   - 沿光线每轮取最近的 16 个交点，按 alpha 从前往后混合 c_ref = Σα_i·T_i·c_i，直到 T < 1e-4；
   - **`START_OFFSET 0.06`**：光线参数 t < 0.06（以 |ref_d| 为单位，实际 6 到约 9.5 cm）的命中跳过，避免打到镜子自己的面元。它是 ptx 常数，改它要另编一份（`build_variants/`）；
   - 背景是黑色；
   - **共用版**：c_i = SH2RGB(DC)，**和视角无关**。
6. **合成**：追踪了的像素 c = (1 − β)·c_base + β·c_ref，没追踪的 c = c_base。
   可视化里 DIFFUSE = (1 − β)·c_base，REFLECTION = β·c_ref。
7. **梯度**：被反射光线命中的面元，DC 和几何都收到梯度。
   - 镜面法线 / 曲面通过 ∂c_ref/∂ref_d 收到梯度，实质是"镜中画面的空间梯度"，捕获范围有限：转 1°，2–3 m 外的命中点就移动 7–10 cm；
   - 起点通过 ∂c_ref/∂ref_o 把梯度回传到镜面深度；
   - 增密统计只来自相机视角的栅格化。

## 2. 训练中的特殊操作（`update_dif_gaussians`，只在 iter < 21000）

```
0      500        3000             6000  9000  12000  15000  18000     21000        45000       60000
|------|----------|----------------|-----|-----|------|------|---------|------------|-----------|
       增密开始     反射开始、opacity reset 每 3000                          增密/重置结束  LPIPS 开始
                  β reset：6000 起每 3000（5 次）
                  每 1000：color sabotage + normal propagation（3000 < iter ≤ 18000）
bfilt：0–3000 不追；3000–4500 全追；之后筛选，但每次 β reset 后 1500 次内全追；19500 以后一直筛选
msurf：4500 起曲面可学；mirror_gs_norm 和一致约束 3000 起
```

1. **增密与剪枝**（500–21000）：
   - 间隔：< 3000 时每 100 次，3000–18000 每 500 次，18000–21000 每 100 次。
   - 梯度 ≥ 2e-4 时：尺度 ≤ 0.01 × spatial_scale 的 clone，更大的 split。
   - opacity < 0.05 的剪掉。
   - 尺度 > 0.1 × spatial_scale 的面元：累计权重低于 10% 分位的删掉，否则分成 5 个。
   - 超过 180 万个时，按可见度剪到 180 万。
2. **Opacity reset**：每 3000 次，opacity ← min(opacity, 0.01)。共用版里反射到的场景会同时变暗。
3. **β reset**：6000、9000 … 18000 这 5 次，β ← min(β, 0.001)（clip）。
   试过 ×0.1 软 reset（`reset_specular_scale`，bsoft 组），没有收益。
   betalog 实测：reset 后约 500 次 β 回到 0.5，约 1000 次回到原来的水平。
4. **Color sabotage / normal propagation**（来自 3DGS-DR，每 1000 次）：
   - β ≤ 0.05 的面元 DC 加噪声 U(−0.4, 0.4)；
   - 所有面元 opacity ≥ 0.9，β ≥ 0.02 的面元尺度 × 1.5，opacity 冻结 200 次。
   - 这两个操作都依赖 β 阈值。
5. **bfilt：只追 β 大的像素**（`specular_filtering_*`）：
   - 先栅格化出 β 图，只对 β > 0.02 的像素追反射；整张图都没有就保底追 β 最大的一个像素。
   - 每次 β reset 后 1500 次内全部追踪，否则 β 恢复不了。
   - r300lp 上后期只追约 14% 的像素，快 2.3–2.7 倍，画质不变。验证和测试渲染同样筛选。
   - 追踪占比记在 `TRAIN/ref_frac`。
6. **SH 阶数**：每 1000 次 +1，最高 3 阶。

## 3. Loss

总 loss 是各项加权求和。**像素级 loss 一律对整张图的全部像素取平均。**

| loss | 权重（主线） | 起始 | 公式 | 代码 |
|---|---|---|---|---|
| RGB L1 | 0.8 | 0 | mean\|c − c_gt\| | VolumetricVideoSupervisor |
| D-SSIM | 0.2 | 0 | 1 − SSIM | 同上 |
| LPIPS-VGG | 0.01 | 45000 | | 同上 |
| 法线先验 | 0.05 | 0 | [‖n_v − n_p‖₁ + (1 − cos)] × w_dpt；先验为黑的像素跳过（`norm_ignore_zero`） | EnvGSSupervisor |
| gs_norm | 0.04 | 0 | [1 − n·N_d] × w_dpt | 同上 |
| 镜面 gs_norm | 0.5 | 3000 | 掩码 255 内 1 − n_render·n_depth，不乘 w_dpt | MirrorRegionSupervisor |
| LiDAR 深度 | 0.05 | 0 | LiDAR 落点像素上 max(\|z_render − z_lidar\| − 0.05, 0) | LidarDepthSupervisor |
| 曲面锚定 | 0.01 | 4500 | LiDAR 内部格点到曲面的 Huber(残差 / 2 mm) | MirrorSurface |
| 曲面正则 | 1e-4 | 4500 | B 样条二阶差分相对初值变化的均方 | SplineSurface |
| 一致：深度 | 1.0 | 3000 | 掩码 255 且命中曲面：L1(dpt_map, t) | MirrorRegionSupervisor |
| 一致：法线 | 0.5 | 3000 | 同上像素：1 − \|n_render · n_s\| | 同上 |

w_dpt = 1 − (d − d_1%)/(d_99% − d_1%)，截到 [0, 1]：近处权重大，最远 1% 的权重为 0。

**各项的作用和来历**（详细数据见 `RESULTS.md`、`archive/`）：
- **法线先验**：没有它（或者镜面用 StableNormal），平面镜会退化成"镜后虚像房间"。有 msurf 后，它主要约束面元的朝向。
- **镜面 gs_norm**：压镜面上的鼓包。
  - 权重 1.0 会让镜面面元伸出镜缘 0–5 cm，墙面掉 5 dB；
  - f300 扫描 0.1 / 0.2 / 0.5 后，用户定 0.5。
- **LiDAR 深度**：几何几乎全对（天花板 4.6 → 0.3 cm），也能去掉镜后 floater；镜面画质基本不变。
  - 5 cm 死区（约 2σ）让测距噪声不再拉扯，画质在 LiDAR 组里最好；
  - 权重 0.01–0.1 都能修好天花板，几何随权重单调变好。
- **一致约束**：把右侧比曲面靠前 4–5 cm 的高斯层拉回曲面。c2_480 镜面 27.89 → 28.71，右侧次外带 +4.3 dB，曲面法线 p90 从 0.32° 降到 0.16°。
  - 镜面内加 2DGS distortion（权重 10）反而掉分，不用。

**代码里有、但主线没开的项**：
- `gs_dist_loss_weight`：全图 2DGS distortion。
  - 实现已验证正确（`check_dist_loss.py`）。它算的是归一化深度的平方差，2–3 m 处相隔 5 cm 的两层只贡献约 3e-6；
  - r300lp 上权重 100 掉 2–5 dB，权重 10 的不透明区掉 5 dB。
- `mirror_beta_loss_weight`：逐像素把镜面 β 往 0.9 拉。c480 上会长出满屏的相机近处浮游物，崩了。
- `mirror_norm_loss_weight`、`mirror_rgb_scale`、`lidar_dpt_loss_weight_final`（ldecay）、`reset_specular_scale`、`ref_ss`（反射多采样）：
  都试过，没有收益或者没进主线。
- 原版就有、但没实现或用不上的：`env_opacity`、`msk`、`norm_smooth`、`specular_loss`、`ref_rgb_loss`、`res_norm_loss_weight`、`use_spec_scale_*`。

## 4. 已知问题

- **共用版反射颜色只有 DC**：镜中画面没有视角相关效果，方向相关的梯度为零；β 和 c_ref 可以此消彼长，**β 不能读成反射比例**，要看反射占比 β·c_ref / c。
- **每像素一条反射光线**：没有抗锯齿。镜子在画面里越小越伤，走廊里镜面每像素约 4 mm，房间约 1.6 mm。`ref_ss` 多采样在推理上有效，没进训练主线。
- **增密统计只来自相机视角**：镜中看到的物体不会因为反射而增密。
- **镜面指标末段波动**：训练末段镜面指标随迭代来回波动，约 LPIPS ±0.01、PSNR ±0.8 dB（r300lp 实测）。单次运行之间小于约 1 dB 的差别不算定论。
- **走廊场景视角覆盖差**：镜子挂在墙角，基本只在镜前拍。c480 用完美法线上限也只有 27.6（房间 36.8）。
- **镜缘**：范围没覆盖到的地方回退到 2DGS 法线；tracer 的 START_OFFSET 会跳过镜框；B 样条镜缘前期变差。见 `MIRROR_SURFACE.md`。
