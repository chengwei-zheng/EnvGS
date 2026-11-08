# EnvGS 训练流程：所有特殊操作与 loss（2026-10-22 整理，10-25 补充渲染细节和梯度分析）

以当前主线 `configs/exps/envgs/mirror_b/envgs_r300lp_lidarnorm_shared_bfilt.yaml` 为准，数值都从代码和配置里核对过。
- 数据：`dataset_r300_lamppanel`（r300lp）。和 r300 的相机轨迹、训练/验证划分相同，区别是灯换成了发光面板：r300 的面光源相机看不见、镜子里却看得见，是个 bug。
- 和 10-22 版的主线 `envgs_r300_lidarnorm_shared` 相比，只多了 bfilt（§1.7）。
- 有变体的地方单独标出来（原版 / ldepth / linit / bsoft 等）。
- **2026-10-28 用户选定的当前最好设置：`envgs_r300lp_lidarnorm_shared_bfilt_mgsn02.yaml`** = 主线 + 镜面掩码内额外的深度-法线一致项
  （`MirrorRegionSupervisor` 的 `mirror_gs_norm_loss_weight: 0.2`、`mirror_gs_norm_start_iter: 3000`；逐像素 1 − n_render·n_depth，
  只在 LiDAR 镜面掩码 255 内、不乘深度缩放、对全部像素取平均）。镜面鼓包 >10° 从 1.5% 降到 0.29%，镜面视觉最好；代价是不透明区
  PSNR −1.8 dB、镜面平面上的面元伸出镜边 0–1.6 cm。权重 1.0（`_mgsn1`）鼓包更少，但镜面伸出镜边 0–5 cm、墙面 −5 dB（NEXT_STEPS 10-27）。

- **2026-11-05 房间场景最终方法（用户确认）：可学 B 样条显式镜面 + LiDAR 第一回波深度**，平面镜
  `configs/exps/envgs/mirror_b/envgs_f300_msurf_spline_learn_ldepth005_dz5_fitfix.yaml`（镜面 PSNR 38.64，基线 28.56），凸面镜
  `envgs_cv300_msurf_spline_learn_ldepth005_dz5.yaml`（34.63，基线 24.45）。= f300 默认（`_mgsn05`）+ `sampler_cfg.mirror_surface_cfg`
  （spline，2 cm 控制点，LiDAR 全局二次曲面拟合初始化，4500 次起可学，lr `msurf_h` 2e-6）+ `LidarDepthSupervisor`（0.05，5 cm 死区）。
  镜面像素（相机光线打到曲面、在渲染深度 tol 内）的反射法线换成曲面法线，反射起点仍是渲染深度。结论和对照表见 NEXT_STEPS.md 开头；
  viewer：`mirror_lidar/tools/gui_f300_msurf_best.sh`、`gui_cv300_msurf_best.sh`。

- 代码位置
  - 采样器：`easyvolcap/models/samplers/envgs_sampler.py`（继承 `gaussian2d_sampler.py`）
  - 高斯模型的各种操作：`easyvolcap/utils/gaussian2d_utils.py`
  - loss：`easyvolcap/models/supervisors/{volumetric_video,envgs,lidar_depth,mirror_region}_supervisor.py`
- 配置继承链：`configs/base.yaml` → `configs/models/envgs.yaml` → `configs/datasets/mirror_b/{mirror_b,r300_lidarnorm}.yaml` → 实验 yaml

---

## 0. 基本设置

| 项 | 值 |
|---|---|
| 总迭代 | `epochs 120 × ep_iter 500 = 60000`，batch 1，每次一张完整图（`n_rays -1`） |
| 分辨率 | `ratio 0.5` → 1296×972 |
| 训练 / 验证视角 | 262 / 38（每 8 张取 1 张做验证） |
| 初始化 | 主面元：`sparse/0/points3D.ply`（COLMAP 用 GT 位姿三角化，只用 RGB）；环境高斯：`envs/points3D.ply` |
| 场景尺度 | `spatial_scale 1.993`，凡是带 × spatial_scale 的阈值都乘它 |
| 优化器 | Adam，eps 1e-15；学习率：xyz 1.6e-4 × spatial_scale 指数衰减到 1/100（3 万次衰减完，之后保持）；DC 2.5e-3，SH rest 1.25e-4，opacity 0.05，scaling 5e-3，rotation 1e-3，specular(β) 0.01 |
| 法线先验 | `normals/`：StableNormal 单目法线；镜面内部替换为 LiDAR 多回波拟合的平面法线，镜子边缘一圈存成黑色（不监督） |

### 初始化（核对过代码和数据，2026-10-22）
- **主面元**：`r300/sparse/0/points3D.ply`，共 171,745 个点；r300lp 是 `r300lp/sparse/0/points3D.ply`，共 172,570 个点，生成方法相同。
  - 由 `~/SMBC/EnvGS/tools/blender_gt_to_colmap.py` 生成：GT 位姿固定，COLMAP `point_triangulator` 只三角化，只用 RGB，Blender 世界坐标（米，Z 朝上）。
  - 范围约 x −26 到 17、y −18 到 34、z −28 到 12 米，远大于房间，有离群点。
  - 参数（`GaussianModel.create_from_pcd`）：
    - DC = rgb2sh0(点的颜色)，SH 高阶全 0（SH 阶数从 0 开始）；
    - 两轴尺度相同，都等于到最近 3 个点的均方根距离（`distCUDA2`，中位数约 1.4 cm）；
    - 旋转 = normalize(torch.rand(4))：随机朝向，而且不是均匀的随机旋转；
    - opacity 0.1，β 0.001，roughness 0.5（roughness 不用）。
- **LiDAR 点云初始化（linit 组，2026-10-23）**：数据集换成 `r300lp_lidarnorm_linit`，只替换 `sparse/0/points3D.ply`，其余文件是符号链接。
  - 生成脚本：`mirror_lidar/tools/lidar_init_points.py`。
  - 取 262 个训练帧的第一回波，转到世界坐标后投到同一帧的相机里取颜色；画面外的点和被挡住的点不要（8 px 小块 z-buffer，5 cm 容差）。
  - 按 4 cm 体素下采样：颜色取格内均值，位置取周围 3×3×3 格内所有原始点的质心，用来抵消沿射线方向的测距噪声。
  - 结果 295,415 个点：中位误差 0.91 cm，离表面超过 5 cm 的 2.1%，镜中虚像 0.14%；COLMAP 分别是 0.61 cm、11.9%、8.25%。
  - 后面进 `create_from_pcd` 的流程和上面完全一样，旋转仍然随机。
- **环境高斯**（只在原版里用）：`envs/points3D.ply`，163,840 = 32³ × 5 个点。
  - 训练第一次启动时由 `init_env_points` 生成：在 env_bounds 内切 32³ 个格子，每格随机撒 5 个点，颜色随机；opacity 0.1，3 阶 SH。
- **spatial_scale 和 env_bounds**：由 `scripts/preprocess/tools/compute_metadata.py` 算出。
  - spatial_scale = 训练相机中心离它们平均位置的最远距离 × 1.1 = 1.993；
  - env_bounds = COLMAP 点云坐标的 0.5%–99.5% 分位数构成的盒子。
- **DC**：SH 的第 0 阶系数（不随视角变化的颜色），RGB = 0.282·DC + 0.5。共用版的反射只取被命中面元的 DC。

### 渲染步骤（每次迭代，`EnvGSSampler.forward`，2026-10-25 逐行核对代码）

**结论先说：每个像素只有一条相机光线和一条反射光线。** 反射方向是理想镜面方向，不做粗糙度采样（`multi_sampling_start_iter −1`，roughness 不参与渲染），也没有二次反射（`max_trace_depth 0`）。但一条反射光线并不是只取第一个命中点，而是像栅格化一样，把沿途的多个面元按 alpha 从前往后混合。

1. **训练期操作**：`update_dif_gaussians`，包括增密、剪枝和各种重置，见 §1。
2. **相机光线**：每个像素一条，`get_rays` 取像素中心（`correct_pix`）。
   - o 是相机中心。
   - d 按 **z 深度约定**取：`use_z_depth True`（`gaussian2d_sampler.py:68`），d 的相机系 z 分量为 1，所以 |d| = 1/cos θ（r300 图像边缘 θ 约 50°，|d| 最大约 1.6）。
   - 这样 o + d·z 就是 z 深度为 z 的那个点。10-25 实测，镜面像素的 o + d·dpt 到镜面平面的距离中位数约 0.3 cm。
3. **主面元栅格化**（2DGS，`diff_surfel_rasterization`，`gaussian2d_utils.py` 的 `render`）：
   - 每个像素和面元求交：先算光线与面元平面的交点 uv，权重 G = exp(−½|uv|²)，和 2D 低通滤波取较大者。
   - 不透明度 α = min(0.99, opacity·G)，从前往后混合，权重 w_i = α_i·T_i。
   - 输出的各张图：
     - `rgb_map` = c_base：SH 按视角方向求值后 +0.5，截断到 ≥0。SH 阶数从 0 开始，每 1000 次 +1，最高 3 阶。
     - `acc_map` = Σw。
     - `dpt_map`：期望 z 深度 Σw·z / Σw（`depth_ratio 0`，不用中值深度）。
     - `norm_map`：Σw·n，面元法线朝向相机一侧，转到世界坐标，没有除以 acc，用的时候再归一化。
     - `surf_norm_map`：由 `dpt_map` 有限差分得到的法线，再乘 acc。
     - `spec_map`：Σw·β，β = sigmoid(_specular)，初始 0.001。
4. **反射光线**（`get_reflect_rays`），iter ≥ 3000（`render_reflection_start_iter`）时才有：
   - 起点 `ref_o = o + d·dpt_map`，即渲染出的表面点。
   - 方向 `ref_d = d − 2(d·n)n`，n = normalize(norm_map)。没有归一化，所以 |ref_d| = |d|，这和 tracer 的 z 深度约定一致。
   - `detach False`：起点和方向都保留梯度。
   - bfilt 组在筛选期间只取 β > 0.02 的像素，见 §1.7。
5. **OptiX 追踪**（`diff_surfel_tracing`，`optix_tracer/forward.cu`）：
   - 每个面元在 BVH 里是一个 ±3σ 的四边形，即两个三角形（`OptiXTracer.get_disks`）。BVH 每次训练前向都重建（`build_bvh(rebuild=self.training)`），所以面元移动后马上生效。
   - 沿光线每轮用 any-hit 收集最近的 16 个交点（`CHUNK_SIZE`），按距离排序，逐个处理：
     - 在面元平面上算 uv，G = exp(−½|uv|²)，α = min(0.99, opacity·G)；
     - α < 1/255 的跳过；
     - 光线参数 t < `START_OFFSET 0.06` 的跳过。t 以 |ref_d| 为单位，所以实际是 6 到约 9.5 cm，用来避免打到镜子自己的面元；
     - 面元双面可见（`DUAL_VISIABLE 1`）。
   - 从前往后混合 c_ref = Σ α_i·T_i·c_i，直到 T < 1e-4，或交点超过 256 个（`MAX_INTERSECTION`），或这一轮不满 16 个交点（说明后面已经没有面元）。没混满的部分 T 乘背景色，`env_bg_brightness 0`，也就是黑色。
   - c_i 的来源：
     - **共用版（`shared_reflection True`，10-21 起是默认值）**：追踪主面元 `self.pcd` 本身，`override_color` = SH2RGB(DC) 截断到 ≥0，**和视角无关**。
     - **原版（`shared_reflection False`）**：追踪一套独立的环境高斯，3 阶 SH，按反射方向求值。它们有自己的增密和剪枝（到 21000，间隔 500）和不透明度重置（每 6000 次）。共用版里环境高斯不参与，也不做增密和重置。
   - tracer 也输出反射光线上的深度和法线，但目前没有 loss 用到。
6. **颜色合成**（`store_env_gaussian_output`）：
   - 追踪了的像素：**c = (1 − β)·c_base + β·c_ref**，β 是 `spec_map`。
   - 没追踪的像素（只有 bfilt 筛选期间才有）：c = c_base。
   - 可视化里 DIFFUSE = (1 − β)·c_base，REFLECTION = β·c_ref，RENDER = 两者之和。
7. **loss 和反传**：见 §2。
   - 被反射光线命中的面元，DC 颜色和几何参数都会收到梯度。
   - 增密统计（视空间梯度、累计权重）只来自相机视角的栅格化。tracer 里虽然也有 `a_weights` 累加，但共用版没有用到。

### 颜色怎么影响镜面法向（2026-10-25，`mirror_lidar/tools/mirror_normal_grads.py`）
- **梯度路径**：∂c/∂n = β · ∂c_ref/∂ref_d · ∂ref_d/∂n，其中 ∂ref_d/∂n = −2[(d·n)I + n dᵀ]。
  - **∂c_ref/∂ref_d 的含义**：反射方向一变，沿途每个命中面元上的交点 uv 跟着移动，G 随之变化，混合出的颜色就变了。实质上就是"镜中画面在这一点的空间梯度"，和光流、图像配准是同一类信号。
  - **没有梯度的部分**：
    - 命中哪些面元、按什么顺序命中，是离散的，不可导；
    - 共用版颜色只取 DC，所以 SH 随方向变化的那部分梯度为零。
  - **起点的梯度**：起点 ref_o 依赖深度，所以颜色误差也会回传到镜面深度。
- **捕获范围有限**：镜面转 δ，反射光线转 2δ；在距离 L 处，命中点移动 2δL。r300 里 L 约 2–3 m，转 1° 时命中点移动 7–10 cm。渲染的镜中画面和真实画面错开太多以后，局部图像梯度就不再指向正确方向。
- **实测**：训练好的 bfilt，镜面整体倾斜 δ，每项 loss 的回复斜率 dL/dδ（1/rad，> 0 表示把镜面拉回原位）：

  | 倾角 | L1 | SSIM | LPIPS | 法线先验 | gs_norm | LiDAR 深度（ldepth 组） |
  |---|---|---|---|---|---|---|
  | 2° | +0.118 | +0.189 | +0.034 | +0.017 | +0.001 | +0.019 |
  | 5° | +0.077 | +0.010 | +0.002 | +0.016 | +0.001 | +0.018 |
  | 10° | +0.043 | +0.032 | +0.003 | +0.012 | +0.001 | +0.018 |
  | 20° | +0.009 | +0.032 | +0.002 | +0.013 | −0.000 | +0.018 |

  - **偏差小时颜色主导**：2° 时颜色约是法线先验的 10 倍；20° 时 L1 接近 0，在 linit 组甚至为负。
  - **法线先验和 LiDAR 深度在任何角度都稳定**：大小几乎不随倾角变化，是"全局"的回复力。其中 LiDAR 深度只约束镜面位置，镜面面元原地转动时它约等于 0。
  - **gs_norm 可以忽略**。
  - 局限：这是在已经收敛的模型上测的。训练早期镜中画面还没学好，颜色没有正确的目标；这也解释了 r300 上不加法线先验时会收敛到"镜后虚像房间"。

---

## 1. 训练中的特殊操作（都在 `update_dif_gaussians`，每次迭代前执行，只在 iter < 21000 时运行）

时间线（迭代数）：

```
0      500        3000             6000  9000  12000  15000  18000     21000                 60000
|------|----------|----------------|-----|-----|------|------|---------|---------------------|
       增密开始     反射开始                                      ↑停止 color sabotage / normal prop
                  每 3000 次: opacity reset（3000 起）
                             每 3000 次: β reset（6000 起，5 次）
                  每 1000 次: color sabotage + normal propagation（3000 < iter ≤ 18000）
                                                                          ↑增密/剪枝/所有重置结束
                                                                                   45000: 开始 LPIPS
bfilt 组（§1.7），全 = 每个像素都追反射，筛 = 只追 β > 0.02 的像素：
0      3000 4500 6000 7500 9000 10500 12000 13500 15000 16500 18000 19500                  60000
|-不追-|-全-|-筛-|-全-|-筛-|-全-|--筛-|--全-|--筛-|--全-|--筛-|--全-|----------筛----------|
```

### 1.1 增密与剪枝（2DGS / 3DGS 标准做法 + EasyVolcap 改动）
- 时间：iter > 500 开始，到 21000 结束。
- 间隔：iter < 3000 时每 100 次；3000–18000 每 500 次；18000–21000 又回到每 100 次。
- 每次的顺序：
  1. **clone**：视空间位置梯度均值 ≥ 2e-4 且最大尺度 ≤ 0.01 × spatial_scale 的面元，复制一份。
  2. **split**：梯度条件相同、尺度 > 0.01 × spatial_scale 的面元，分裂成 2 个。
  3. **按不透明度剪枝**：opacity < 0.05 的面元删掉。
  4. **大面元处理**（`prune_large_gs True`）：最大尺度 > 0.1 × spatial_scale 的面元，
     - 若累计权重低于全体的 10% 分位（`min_weight_threshold 0.1`），删掉；
     - 否则分裂成 5 个、尺度 × 0.5。
  5. **按可见度剪枝**（`prune_visibility True`）：点数超过 `max_gs × 0.9 = 180 万` 时，按累计权重从小删到 180 万。现在只有约 50 万，没有触发。
- 屏幕尺寸相关的条件（`split_screen_threshold`、`max_screen_threshold`）都是 None，没有启用。

### 1.2 Opacity reset
- 每 3000 次（3000、6000 … 18000；21000 时增密模块已停，不再触发），所有面元 opacity ← min(opacity, 0.01)，Adam 动量清零。
- 之后低不透明度的面元在下一次剪枝时被删掉，用来清理漂浮物。

### 1.3 β（specular）reset
- 和 opacity reset 在同一次迭代，额外要求 iter > 3000，所以实际在 **6000、9000、12000、15000、18000** 这 5 次。
- **默认（clip）**：β ← min(β, 0.001)，所有大于 0.001 的 β 压回 0.001，Adam 动量清零。
- `reset_specular_all True`（没用）：所有面元的 β 统一设成 0.001。
- **`reset_specular_scale > 0`**（10-22 新加，bsoft 组用 0.1）：β ← β × scale，保留面元之间的高低顺序。
- 目的：让每个面元重新"争取"自己是不是反射面，压掉靠 β 补颜色的假反射。18000 次以后 β 不再受约束，自由收敛 42000 次。
- 共用版的副作用：reset 的同时主面元 opacity 也被压到 0.01，反射光线追到的场景短时间内也是半透明、偏暗的。

### 1.4 Color sabotage（`distort_color`，来自 3DGS-DR）
- 每 1000 次，在 3000 < iter ≤ 18000 之间，跳过和 opacity reset 重合的那次。
- 对 **β ≤ 0.05 的面元**，DC 颜色加均匀噪声 U(−0.4, 0.4)（SH 空间）；β > 0.05 的面元不动。
- 目的：非反射面元的颜色被打乱后要重新拟合；如果某块区域其实是反射，用 β 解释会更省力，这样能把反射"逼"给 β。

### 1.5 Normal propagation（`enlarge_opacity` + `enlarge_scaling`，来自 3DGS-DR）
- 和 color sabotage 同一个时间点，即每 1000 次（3000 < iter ≤ 18000，跳过 reset 那次）。
- 所有面元 opacity ← max(opacity, 0.9)。
- **β ≥ 0.02 的面元**尺度 × 1.5；β < 0.02 的不动。
- 随后把 opacity 的学习率设成 0；每 200 次（`opacity_lr0_interval`，同一区间内）恢复成 0.05。
  - 设成 0 的代码在恢复的代码之后，所以效果是每次 normal propagation 之后，opacity 冻结 200 次迭代。
  - 18000 那次不冻结。
- 目的：反射面元变大、变不透明，好的法线能覆盖到相邻区域。

### 1.6 其它
- SH 阶数从 0 开始，每 1000 次 +1，到 3 阶为止（到 21000 之前都会检查，实际 3000 次时已经是 3 阶）。
- 环境高斯（只在原版里用）：增密从 500 次到 21000 次，间隔 500，梯度阈值 1e-4，每 6000 次 opacity reset。共用版里全部跳过。

### 1.7 只追踪 β 大的像素（bfilt 组，2026-10-22 新加）
- 配置：`envgs_r300lp_lidarnorm_shared_bfilt.yaml`。sampler 参数如下：
  - `specular_filtering_start_iter 4500`
  - `specular_filtering_threshold 0.02`
  - `specular_filtering_pause_iter 1500`
- 代码：`EnvGSSampler.is_specular_filtering`、`get_reflect_rays`。
- **筛选时**：先栅格化得到 β 图（spec_map），只对 β > 0.02 的像素做反射追踪，这些像素的颜色是 c = (1 − β)·c_base + β·c_ref；其余像素不追踪，颜色直接用 c_base。
  - 如果整张图都没有超过阈值的像素，就保底追 β 最大的那一个，因为 tracer 需要至少一条光线。
- **作用区间**：
  - 0–3000：还没开始反射，不追踪。
  - 3000–4500：反射刚开始，β 全是 0.001，每个像素都追。
  - 每次 β reset（6000、9000、12000、15000、18000）后的 1500 次内每个像素都追，即 6000–7500、9000–10500、12000–13500、15000–16500、18000–19500。reset 后 β 全被压到约 0.001，这时候筛选，镜面像素也会被筛掉，β 没有梯度、就涨不回来了。betalog 显示 reset 后约 1000 次 β 才恢复，所以取 3000 次重置间隔的一半，1500 次。
  - 其余时间都筛选：4500–6000、7500–9000、10500–12000、13500–15000、16500–18000，以及 19500 到训练结束（60000）。
  - 验证和测试渲染时 iter 等于当前迭代数，所以同样会筛选。
- 效果（r300lp）：后期只追约 14% 的像素，速度 6577 次/小时，基准是约 2450 次/小时，快 2.7 倍；6 万次一共用 9 小时。
- 被追踪像素的占比记在 tensorboard `TRAIN/ref_frac`（MirrorRegionSupervisor `log_mirror_stats`），不筛选时记 1。

---

## 2. Loss

总 loss = 下面各项加权求和。像素级的 loss 都是在整张图上取平均（mean over all pixels）。

| loss | 权重 | 起止 | 公式 | 代码 |
|---|---|---|---|---|
| RGB L1 | 0.8 | 全程 | mean \|c − c_gt\| | VolumetricVideoSupervisor（`img_loss_type L1`，不是 Huber） |
| D-SSIM | 0.2 | 全程 | 1 − SSIM(c, c_gt)，11×11 窗口 | 同上 |
| LPIPS | 0.01 | iter > 45000 | LPIPS-VGG(c, c_gt) | 同上（`perc_loss_start_iter 45000`，原模型配置是 21000，我们为了加速改晚了） |
| 法线先验 | **0.05**（模型默认 0.01） | 全程 | [‖n_v − n_p‖₁ + (1 − cos(n_v, n_p))] × w_dpt | EnvGSSupervisor `norm_loss` |
| 面元法线一致性 gs_norm | 0.04 | 全程 | [1 − n · N_d] × w_dpt | EnvGSSupervisor `gs_norm_loss` |
| LiDAR 深度（只在 ldepth 组） | 0.1 | 全程 | 在 LiDAR 落点像素上算 mean \|z_render − z_lidar\| | LidarDepthSupervisor |

各项的细节：

- **法线先验**
  - n_v = normalize(norm_map) 转到相机坐标系（`@ R^T`）；n_p = 先验图 × 2 − 1 再归一化。
  - `norm_ignore_zero True`：先验为黑色（镜子边缘一圈）的像素不算。
  - w_dpt = `normalize_depth(dpt.detach())` = 1 − (d − d_1%) / (d_99% − d_1%)，截到 [0, 1]：越近的像素权重越大，最远的 1% 权重为 0。
  - 镜面内部用的是 LiDAR 拟合的法线（lidarnorm），或 GT 法线（gtnorm 组）。
- **gs_norm**
  - n 是 norm_map（面元法线，NORMAL），N_d 是 surf_norm_map（深度有限差分法线，SURFACE_NORMAL）。同样乘 w_dpt。
  - 作用是让面元的朝向和它们形成的表面一致（2DGS 的 normal consistency）。
- **LiDAR 深度**
  - 只用第一回波：每个区域取距离最近的回波，不区分回波类型。每张图约 3.9 万个点（3.1% 像素）。数据在 `data/datasets/mirror_b/r300_lidar_depth`，r300lp 用 `r300lp_lidar_depth`（中位误差 1.62 cm）。生成脚本 `mirror_lidar/tools/lidar_first_echo_depth.py`。
  - 训练中这项 loss 从约 2000 次起就停在 2.4 cm 左右。测距噪声 σ ≈ 2.4 cm，对应的 L1 下限约 1.9 cm，所以几何很早就拟合到了噪声水平。
  - 可选 `lidar_dpt_loss_weight_final` 和 `lidar_dpt_loss_decay_iter`，在某次迭代后换成另一个权重。r300lp 的 ldecay 组（21000 次起 0.02）每项指标都不如恒定 0.1。
  - 权重扫描（linit + ldepth，0.1 / 0.03 / 0.01）：0.01 就能把天花板修好（0.8 cm）；几何随权重单调变好，0.1 最好。
- **MirrorRegionSupervisor**（`mirror_lidar/tools/lidar_mirror_mask.py` 生成 LiDAR 镜面掩码，255 镜面内部 / 128 边缘一圈 / 0 其它）
  - `mirror_norm_loss_weight`：镜面内部额外加一份法线先验 loss（nsplit 组 0.04，10-21 停掉了，没有结果）。
  - `mirror_rgb_scale` + `mirror_rgb_until_iter`：在指定迭代前，把镜面区域（内部 + 边缘）的 L1 和 SSIM 乘一个系数（mrgb 组 0.1 到 5000 次，10-21 停掉了，没有结果）。
  - `log_mirror_stats`：只记录、不加 loss。记镜面上 β 的均值、反射占比（β·c_ref / c）、c_ref、非镜面区域 β 的均值，以及被追踪像素的占比 `ref_frac`。r300lp 各组都开着。

代码里有、但我们的配置里**没有启用**（权重为 0 或没设置）的项：
- **gs_dist**（`gs_dist_loss_weight`，`configs/models/envgs.yaml` 里显式写 0，注释 "do not use this"）：就是 2DGS 的 depth distortion loss。
  - 光栅化器（`diff-surfel-rasterization-wet-ch05`）输出 `rend_dist`（allmap 第 7 通道），loss 取全图均值，默认从 3000 次开始。
  - 和 2DGS 官方代码一致，实际算的是 Σ_{j<i} w_i w_j (m_i − m_j)²：不是论文写的 |z_i − z_j|，而是归一化深度 m(z) = far/(far−near)·(1 − near/z)（near 0.2、far 100）的平方差。dm/dz ≈ 0.2/z²，所以 2–3 m 处相隔 5 cm 的两层只贡献约 3e-6，权重得是几百到上千才有作用；bfilt 训练末期全图均值约 1.3e-4。`DETACH_WEIGHT 0`：梯度也会流到权重。
  - 10-27 已验证实现正确（`mirror_lidar/tools/check_dist_loss.py`）：用 3 个正对相机的面元，中心像素的值和闭式解一致（比值 1.0000）；对深度、不透明度、尺度的梯度和有限差分一致（误差 2% 以内）。源码无本地改动，装的 .so 是 10-12 用这份源码编译的。
  - 它惩罚一条光线上的混合权重分散在不同深度，会把权重压到一个表面上。表面前方的半透明层（不透明区的漂浮层、镜面上的小鼓包）正是它针对的对象。2DGS 原文有界场景用 1000、无界场景用 100。
  - 我们一直没开，所以深度方向上分层，除了 gs_norm 以外没有别的约束。镜面鼓包对 gs_norm 的贡献只有 1–2%（见 NEXT_STEPS 10-27）。
- **env_opacity**（`env_opacity_loss_weight`）：对环境高斯的不透明度加 sparse（log v + log(1−v)，推向 0 或 1）或 L1 正则。但现在的 sampler 不输出 `env_opacity`，所以就算设了权重也不会生效。共用版本来也不用环境高斯。
- **msk**（`msk_loss_weight`）：渲染 alpha 和前景掩码（`batch.msk`，且先验法线非零）之间的 MSE，用于物体级、有前景掩码的数据。室内场景没有掩码，用不上。
- **norm_smooth**（`norm_smooth_loss_weight`，还有 `use_edge_aware_smooth` 等参数）：只有构造函数参数，compute_loss 里没有实现，设了也不生效。
- **specular_loss**（`specular_loss_weight`、`specular_target`、`min_specular_percent`）：只有构造函数参数，没有实现。从参数名看，原意是在 7000–9000 次之间把一部分像素的 β 推向 0.8。
- **ref_rgb_loss**（`ref_rgb_loss_weight`）：只有构造函数参数，没有实现。
- **`use_acc_scale_*`**：norm / gs_norm loss 逐像素乘渲染 alpha（已实现）。**`use_spec_scale_*`**：按 β 缩放，只存了参数，没有实现。我们用的是 `use_dpt_scale_*`，按归一化逆深度缩放。
- **`res_norm_loss_weight`**：构造函数里有默认值 0.001，但 compute_loss 里没有用到。

---

## 3. 各组实验在上面基础上的差别

### r300（旧数据，灯有 bug，只作参考）

| 组 | 反射 | 法线先验（镜面 / 权重） | 额外 | 状态 |
|---|---|---|---|---|
| r300 | 原版 | StableNormal | 权重 0.01 | 完成，镜子变成虚像房间 |
| r300_nmask | 原版 | StableNormal，镜面置黑 | 权重 0.01，`norm_ignore_zero` | 完成，镜子变成虚像房间 |
| r300_gtnorm | 原版 | GT 镜面法线 | 权重 0.01 | 完成，镜面正确 |
| r300_gtnorm_nw5 | 原版 | GT 镜面法线 | 权重 0.05 | 完成 |
| r300_lidarnorm | 原版 | LiDAR 拟合法线 | 权重 0.05，`norm_ignore_zero` | 完成，约等于 gtnorm |
| r300_lidarnorm_ldepth | 原版 | 同上 | + LiDAR 深度 0.1 | 完成 |
| r300_gtnorm_nw5_shared | 共用 | GT 镜面法线 | 权重 0.05 | 完成 |
| r300_lidarnorm_shared | 共用 | LiDAR 拟合法线 | 权重 0.05，`norm_ignore_zero` | 完成 |
| r300_lidarnorm_ldepth_shared | 共用 | 同上 | + LiDAR 深度 0.1 | 停在约 5 万次，没有最终评估 |
| r300_lidarnorm_shared_betalog | 共用 | 同上 | 只多记 β 曲线 | 停在 2.66 万次，β 恢复曲线见 NEXT_STEPS |
| r300_lidarnorm_shared_bsoft | 共用 | 同上 | β reset 改成 ×0.1 | 停在约 2 万次，没有结果 |
| r300_lidarnorm_ldepth_{ldecay,nsplit,mrgb} | 原版 | 同上 | 见 §2 | 10-21 停掉，没有结果 |

### r300lp（新数据，10-22 起，全部是共用版、LiDAR 拟合法线权重 0.05、`norm_ignore_zero`、120 epoch）

| 组 | 额外 | 训练时间 | 结果要点（详见 NEXT_STEPS.md） |
|---|---|---|---|
| r300lp_lidarnorm_shared | 无（基准） | 20.6 h | 全图 PSNR 35.86，镜面 SSIM 0.818 / LPIPS 0.126 |
| _shared_bfilt | bfilt（§1.7） | 9.0 h | 画质持平，不透明区深度更好（中位 2.5 cm）；**推荐默认开** |
| _shared_bsoft | β reset ×0.1 | 21.6 h | 没有收益 |
| _ldepth_shared_bfilt | + LiDAR 深度 0.1 | 10.6 h | 几何最好（F@1 95%，天花板 0.2 cm）；不透明区 PSNR −0.5 dB，镜面 LPIPS 0.138 |
| _ldepth_ldecay_shared_bfilt | + LiDAR 深度 0.1，21000 起 0.02 | 10.8 h | 每项都不如恒定权重 |
| _linit_shared_bfilt | LiDAR 点云初始化 | 11.0 h | 不透明区 LPIPS 最好（0.040）；天花板仍错（5.3 cm）；镜面 LPIPS 0.140 |
| _linit_ldepth{,_w03,_w01}_shared_bfilt | 初始化 + 深度 0.1 / 0.03 / 0.01 | 约 11 h | 几何随权重单调，0.1 和单独 ldepth 相同；两者叠加没有收益；镜面 LPIPS 约 0.15，所有组里最差 |
| _shared_bfilt_mgsn02 | 镜面掩码内额外 gs_norm 0.2，3000 次起 | 约 10 h | 用户 10-28 选为 r300lp 最好：鼓包 >5° 1.5%，镜面 0.800 / 0.153，墙 −2 dB |
| _ldepth_dz5_shared_bfilt_mgsn02 | mgsn02 + LiDAR 深度 0.1、5 cm 死区 | 10.3 h | 全图 35.24 / 0.0812，镜面 0.804 / 0.142，几何 F@2 94%、天花板 0.5 cm；鼓包 >5° 2.0% |

### f300（场景 B_framed：3 cm 厚镜 + 木框，同 r300lp 轨迹；10-28 起，**之后主要在这个场景上做**）

| 组 | 额外 | 结果要点 |
|---|---|---|
| f300_lidarnorm_shared_bfilt_mgsn01 | 镜面 gs_norm 0.1 | 背景最好（全图 LPIPS 0.0730），鼓包 >5° 14% |
| f300_lidarnorm_shared_bfilt_mgsn02 | 0.2 | 镜面 0.761 / 0.169（不单调，疑似单次波动） |
| **f300_lidarnorm_shared_bfilt_mgsn05** | 0.5 | **当前默认（用户 10-29）**：镜面 0.824 / 0.137 / 28.56，鼓包 >5° 2.75%；墙 −0.6 / −1.5 dB |
| f300_lidarnorm_ldepth_dz5_shared_bfilt_mgsn02 | 0.2 + LiDAR 深度 dz5 | 10-29 在跑 |

评估工具（f300 用 `data/logs/eval_f300_lidar.sh`，其余相同）：
- `data/logs/eval_r300lp_lidar.sh <run> <gpu>`：GT 评估、镜面平整度、法线、逐物体统计；
- `scratchpad/pf/mirror_perc.py`：镜面区域的 SSIM / LPIPS；
- `mirror_lidar/tools/geom_eval_b.py`：和 GT 网格比较的几何指标（Chamfer、F-score、法线、面元）。

---

## 4. 已知的问题和值得重新考虑的地方

- **镜面画质随 LiDAR 项变差（10-24，原因未定）**
  - 镜面 LPIPS：bfilt 0.117；加一项 LiDAR（ldepth 或 linit）约 0.14；两项都加约 0.15。镜面 SSIM 没有一致地变差。
  - linit 组的镜面几何确实差一点：镜面向相机偏 1–3 mm，法线中位误差 1.05–1.3°，bfilt 是 0.95°。但 ldepth 组镜面几何没变，LPIPS 照样变差。
  - 下一步：逐帧对比加镜面区域裁剪图。
- **镜面法线主要靠颜色精修，大偏差靠先验**：见 §0 的梯度分析。
  - 法线先验只有 0.05，而且按全图像素平均；0.1° 量级的镜面误差在这项 loss 里几乎没有代价。
  - 要更精确的镜面，可以考虑只在镜面内部加权（`mirror_norm_loss_weight`），或者用 LiDAR 拟合的平面直接约束面元。
- **β reset 太硬**：clip 到 0.001 会丢掉排序信息。r300lp 上试过 ×0.1（bsoft），没有收益，所以保持 clip。
  - betalog 显示 reset 后约 500 次 β 回到 0.5，约 1000 次回到原来的水平；非镜面区域的 β 约 0.004，比镜面低约 100 倍。
- **共用版的反射颜色只有 DC**：
  - 镜中画面没有视角相关的效果，β·c_ref 的颜色由 β 和 DC 一起拟合；
  - 实测 c_ref 约 0.38、β 约 0.75，两者仍然可以此消彼长；
  - 方向相关的梯度也因此为零，见 §0。
- **每像素只有一条反射光线**：没有像素足迹内的抗锯齿，也没有粗糙度波瓣。对镜子够用，但镜中远处的细节会有锯齿和闪烁，这对 LPIPS 不利。
- **反射追踪的速度**：已经用 bfilt 解决，快 2.3–2.7 倍，画质不降。
- **增密统计只来自相机视角**：镜中看到的物体被反射光线命中时不会触发增密，镜中细节只能靠相机直接看到的那部分面元提供。
- **3DGS-DR 的那些技巧都依赖 β 阈值**（0.02 / 0.05）：color sabotage 只打乱 β ≤ 0.05 的面元，normal propagation 只放大 β ≥ 0.02 的面元。β reset 方式一改，这两个阈值的效果也会变。
- **opacity reset 和 β reset 在同一次迭代**：共用版里反射到的场景会同时变暗，变暗的那段时间（reset 后约 1000 次）bfilt 正好每个像素都追。
- **初始点云带镜中虚像**：已经用 LiDAR 初始化（linit）解决，镜中虚像从 8.25% 降到 0.14%。但画质和几何上的收益都被 LiDAR 深度监督盖过去了。
  - 可视化在 `~/SMBC/EnvGS/data/vis/r300lp_init_points/`：`colmap_init_marked.ply` / `lidar_init_marked.ply`，红色为虚像点，橙色为其它离群点。
  - 还没试：用 LiDAR 法线初始化面元朝向（现在旋转是 `torch.rand`，有偏），以及按法线做成扁平的尺度。
- **LiDAR 深度监督的代价**：权重 0.1 时不透明区 PSNR 掉约 0.5 dB；0.01 也能修好天花板，但几何没有 0.1 好。目前还没找到几何和画质兼顾的权重，可以试分区域的权重，或者只在纹理少的区域加。
- **深度误差主要来自低不透明度的漂浮层**：不透明度 > 0.5 的面元中心在每一组都贴着表面（中位 0.12–0.14 cm）。bfilt 里离表面超过 5 cm 的面元占 4.8%，加 LiDAR 深度后降到 1.3–1.6%。
