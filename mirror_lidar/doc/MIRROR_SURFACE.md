# 多回波 LiDAR 镜面拟合与可学镜面曲面（msurf）

更新 2026-11-07。这里只讲方法和用法。命令见 `DATA_PREP.md` 第 4、5 步，结果和对照见 `RESULTS.md`。
总计划见 `~/SMBC/blender/paper_2610/shared_gaussians_lidar_plan.md`。

## 1. 从多回波 LiDAR 拟合镜面

### 输入和原则
只用真机也有的信息：每个 zone 的多回波距离和方向、LiDAR 位姿、相机内外参。
**不用**回波类型标签和强度；合成数据的 `persp/masks` 等真值只用于评测。
合成数据的 `echo_index` 是按强度排序的（镜面上鬼影比表面回波强），所以一律按**距离**重新排序，最近的当第一表面。

### 工具
| 工具 | 镜面形状 | 曲面表达 | 用在 |
|---|---|---|---|
| `tools/lidar_mirror_normals.py` | 平面 | 顺序 RANSAC 平面 | 早期平面镜（r300 / r300lp / f300） |
| `tools/lidar_mirror_normals_curved.py` | 任意光滑曲面（平面也行） | `--surface quadric` 全局隐式二次曲面（**推荐**）；默认 `mls` 局部二次 MLS | cv300、c480、c2_480，以后新数据都用它 |

两者输出格式相同，后面都接 `tools/lidar_mirror_mask.py` 生成掩码。

### 流程（`lidar_mirror_normals_curved.py --surface quadric`）
1. **候选镜面 zone**：同一 zone 有 ≥ 2 个回波；在 256×192 的 zone 网格上做 7×7 开运算，去掉深度边缘混合像素形成的细线；再取连通块，太小的丢掉。
2. **光滑性检查**：每个连通块在自己的 PCA 坐标系下拟合 4 次高度场，要求 MAD < 4 cm、内点 ≥ 70%。
3. **穿透检查**：远回波里有多少落在别的帧直接测到的表面上。镜子约 0.3–0.5，椅背等镂空物体约 1.0，阈值 0.9。
   这一步不需要按镜面翻折鬼影，以后也能用来识别玻璃透射。
4. **镜面片**：所有候选点做 3 cm 体素连通。每片建一个局部系：原点为点的中心，z 轴为 PCA 法线（朝向传感器）。
5. **格子平均**：在 (u, v) 平面上按 1 cm 格子平均回波，权重为回波数。
   - 测距噪声约 2.5 cm。不先平均，直接拟合会把曲率高估 1.6–3.6 倍，代数拟合和 Taubin 拟合都有这个问题，见 `experiments/2026-10-28_quadric_mls_synth/README.md`。
   - 每格约 50 个回波，平均后法向噪声降到约 4 mm。
6. **拟合**：一个隐式二次曲面 f(x) = xᵀAx + bᵀx + c = 0。
   - 10 个系数，9 个自由度。平面、球面、椭球、抛物面、柱面都能精确表达，外推到镜缘也对。
   - Taubin 拟合（min Σw·f² / Σw·|∇f|²，即 10×10 广义特征值问题）。
   - 稳健重加权：按一阶几何距离 |f|/|∇f| 做 Tukey 重加权，阈值 4×MAD，迭代 8 次。
   - **`--rim-drop 0.06`**：离镜缘 6 cm 以内的格子不参与拟合。原因是镜缘 LiDAR 光斑混到镜框，cv300 上离镜缘 0–2 / 2–4 / 4–6 cm 的格点分别往里偏 14.6 / 3.7 / 1.1 mm。
   - 平面镜会退化成 z² ≈ 常数，即两张相距几毫米的平行平面。代码检测到两支根处处相距 < 2 cm 时取中面（11-04 修复）。
     以后新的拟合都要检查高度场 Hh 和斜率 Hu/Hv 一致：有限差分斜率和拟合斜率之差应 < 0.01°。
7. **镜面范围（extent）**：
   1. 在局部系 (x, y) 上取 1 cm 占用格子，统计每格回波密度；
   2. 高斯模糊后，取密度大于内部中位数 1/4 的格子；
   3. 闭运算补洞，取最大连通块；
   4. 求凸包。
8. **投影到每个训练相机**（写法线先验和掩码）：
   - 像素光线和曲面求交（1 cm 高度场 + 牛顿法，带收敛检查），交点要落在范围内，且相机在镜面正面；
   - 遮挡判断：同帧 LiDAR 最近回波明显比交点近（> 5 cm）的是遮挡物，去掉；
   - 掩码开运算、去小块后腐蚀，腐蚀掉的一圈（`--band`，1 个 zone 间距）是边界带：法线存黑色，掩码记 128；
   - 镜面内部的法线换成曲面法线（OpenCV 相机系，存成 (n+1)/2），其余像素保留 StableNormal。
9. **导出**（`--export-ply DIR`）：
   - `patch<i>.npz`：局部系 `c`、`R`，范围网格 `extent`、`g0`、`cell`，采样高度场，二次曲面系数 `quadric` / `fit_scale`，格点 `cells` / `cells_w` / `cells_rim_dist`，以及 `rim_drop`。msurf 读的就是它。
   - `patch<i>_surface.ply`（带法线的网格）、`patch<i>_cells.ply`（颜色 = 离镜缘距离）：可以用 MeshLab 看。
   - `patches.json`。

### 拟合精度
曲面法线误差取中位数，内部 / 外 5 cm 两列都是对 GT 的误差。

| 数据集 | 镜子 | 曲面法线 内部 / 外 5 cm | 掩码精度 / 召回 |
|---|---|---|---|
| r300lp | 平面 | 0.016°（平面版） | 99.96% / 96.8% |
| f300 | 平面 + 木框 | 0.019°（二次曲面修复后） | 99.99% / 92.7% |
| cv300 | 凸面 R 1 m，口径半径 0.54 m | 0.13° / 0.77° | 99.91% / 96.9% |
| c480 | 走廊凸面 R 1 m | 0.16° / 0.71° | 100% / 86.5% |
| c2_480 | 走廊凸面 R 1.125 m，宽 1.35 m | 0.128° / 0.525° | 99.2% / 91.0%（右侧 74%） |

召回没到 100% 的部分，大多是边界带（跳过、不算错）和范围没覆盖到的镜缘。
c2_480 的范围缺两块：顶部约 22 cm（LiDAR 在 1.5 m 高，镜子上缘是掠射，没有回波），以右边缘最多 5.5 cm（轨迹看得少的一侧）。

### 已知局限
- **依赖镜面表面回波**。干净镜面上最近的回波可能就是鬼影，这时拟合不出来。真机正对镜子约 86–90% 的 zone 有表面回波；合成数据 scatter 设为 0.02。
- **镜缘只能外推**，范围也会偏小。
- **标定误差**。曲面是在 LiDAR 系拟合的。cv300 推理敏感性实验：曲面整体转 0.2° / 0.5° / 1° 分别掉 0.3 / 1.5 / 3.5 dB；沿法线平移 5 / 10 / 20 mm 分别掉 0.2 / 1.1 / 2.9 dB。
  真机标定残差常在 0.1–0.3°、5–20 mm，所以曲面位姿需要在训练中可学。
- 真机 LiDAR 和相机不同步，遮挡判断要用时间最近的帧，并做位姿插值。

## 2. 拟合结果的三种用法

1. **法线先验**（`*_lidarnorm*` 数据集）：法线 loss 把 2DGS 面元法线往拟合法线拉（权重 0.05）。
   - 平面镜上有效，能避免"虚像房间"。
   - 曲面镜上面元实际学到的法线仍有 1.4–1.8° 误差（p90 4–5°），反射发糊。
2. **推理时替换反射几何**（只用于检查）：`export_fitted_quadric.py`（任意视角）、`export_fitted_nmap.py`（只在数据集视角）。cv300 上镜面 PSNR 24.45 → 25.79。
3. **训练时显式镜面 msurf（当前方法）**：见第 3 节。cv300 固定曲面就 +6 dB，可学 +10 dB。关键是**从训练一开始就用对的法线**，反射才不会被烘焙进镜面自身的颜色。

命名约定：只用 LiDAR 拟合的叫 `fitted`，用了 GT 的叫 `oracle`。

## 3. 可学镜面曲面 msurf

镜面单独用一个显式曲面表达，从 LiDAR 拟合初始化，训练中可学。**它只替换反射法线**：
- 镜面像素的相机光线和曲面求交，反射方向用交点处的曲面法线；
- **反射起点仍是 2DGS 渲染深度**（`ref_o = o + d·dpt_map`，不 detach）。光度梯度经过起点回传到镜面高斯的深度，所以镜面高斯和反射始终耦合。
  用户 11-06 否决了"起点改用曲面交点"，因为那样反射就和镜面高斯无关了。
- 面元本身、`output.norm_map` 和其它 loss 都不变。

### 代码
- `easyvolcap/models/samplers/mirror_surface.py`
  - `MirrorSurface`（基类）：读 `patch<i>.npz`，负责以下几件事：
    - 范围判断（可用 `extent_dilate` 外扩）；
    - 深度门：交点 t 和渲染深度之差 < `tol`（0.15 m），挡在镜前的物体不替换；
    - 法线朝向；
    - LiDAR 锚定 loss：内部格点（离镜缘 ≥ rim_drop、回波数 ≥ 中位数 1/4），Huber(残差 / 2 mm)；
    - 日志。
  - 子类只实现 `intersect_local`、`normal_local`、`anchor_residual`、`reg_loss`，通过 `SURFACES` 注册。
    - `QuadricSurface`：10 个系数（`msurf_q`）。
    - `SplineSurface`：均匀三次 B 样条高度场，控制点间距 `ctrl`（2 cm），高度 `msurf_h`，从拟合高度场最小二乘初始化。
      正则是控制点二阶差分相对初值变化的均方（曲率单位 1/m）。
  - `targets`（11-06 加）：每像素的交点 t、命中 `hit`（在范围内即可，不看深度门）、曲面法线 n，都 detach，供一致约束使用。
- `envgs_sampler.py`：
  - `mirror_surface_cfg` 为空时完全是原来的行为；
  - `get_reflect_rays` 里，镜面像素（命中且过深度门）的法线换成曲面法线；
  - 输出 `msurf_frac`、`msurf_t` / `msurf_hit` / `msurf_n`。
- `mirror_region_supervisor.py`：
  - 曲面 loss：`msurf_anchor`、`msurf_reg`；
  - 一致约束：`msurf_dpt` / `msurf_norm` / `msurf_dist`；
  - 日志：`msurf_frac`、`msurf_anchor_rms_mm`，有 GT 时还有 `msurf_gt_med` / `msurf_gt_rim5`。

**镜面像素怎么判定**：只看几何，不用 2D 掩码。相机光线要在范围内命中曲面，并且 |t − 渲染深度| < tol。
2D 镜面掩码只用于 loss：`mirror_gs_norm` 和一致约束。

### 配置（当前主线 c2_480）
```yaml
model_cfg.sampler_cfg.mirror_surface_cfg:
    type: spline
    fit: data/result/envgs/corridor/mirror_surface_c2_480_20261104/quadric_rd06/patch0.npz
    learnable: True
    learn_start_iter: 4500      # β 起来以后再学
    ctrl: 0.02
    anchor_weight: 0.01
    reg_weight: 1.0e-4
    extent_dilate: 0.0          # 外扩组 0.04
    eval_sphere: [...]          # GT 球面，只用于日志
runner_cfg.optimizer_cfg.lr_table: {msurf_h: 2.0e-6}   # 二次曲面用 msurf_q: 1.0e-5
```

### 高斯与曲面的一致约束（11-06，`MirrorRegionSupervisor`）
起因：c2_480 学到的曲面几乎就是 GT 球面，但镜子右侧（u < −0.4 m）的高斯渲染深度比曲面靠前 4.3 cm，落在 LiDAR 5 cm 死区内，没有约束。
- **像素**：2D 掩码内部（255）且相机光线在范围内命中曲面（`msurf_hit`）。
- **深度**：`msurf_dpt_loss_weight` 1.0，L1(dpt_map, t)。
- **法线**：`msurf_norm_loss_weight` 0.5，1 − |normalize(norm_map) · n_s|。
- 两项都**对全部像素取平均**（和 gs_norm 一样，每像素权重恒定），曲面一侧 detach（作老师），从 `msurf_cons_start_iter` 3000 开始。
- 可选 `msurf_dist_loss_weight`：镜面内的 2DGS distortion。c2_480 上权重 10 让镜面中部掉 2–3 dB，**不用**。
- 没做 3D 逐高斯 loss：没有准确的 3D 镜面掩码，容易误伤镜框。

### 已知问题
- **B 样条镜缘前期变差**：约 10k 步时镜缘法线有 3.7–4.2° 的峰，20–25k 步后恢复。和 START_OFFSET、深度监督都无关，原因没查清。
- **镜缘 2–5 cm 带最差**（cv300）：反射光线在 tracer 的 `START_OFFSET` 6 cm 内会打到镜框（被跳过）。但重训 2 cm offset 也没有改善，所以这不是主因。
- **镜面上方 0.5–2 cm 有一层高斯**：所有组都有，不是 offset 造成的。
- **范围没覆盖到的镜缘**会回退到 2DGS 法线（c2_480 右侧、顶部），这正是外扩组和 oracle 掩码组要回答的问题。
- **曲面表达能力**：cv300 上可学二次曲面比 B 样条高约 2 dB（模型和球面完全一致）。但 B 样条能表达任意形状，用户 11-03 决定统一用 B 样条。
