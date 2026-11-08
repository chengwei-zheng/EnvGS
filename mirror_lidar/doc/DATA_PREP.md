# 数据预处理：Blender 合成数据 → EnvGS 训练数据（2026-11-07 整理）

以走廊 v2（`dataset_c2_480`）为例，一键脚本 `data/logs/prep_c2_480.sh`。其它场景的脚本结构一样：
`prep_c480.sh` + `prep_c480_mirror.sh`（走廊 v1）、`prep_f300.sh`（B_framed）、`prep_r300lp.sh`（B_rot30 灯面板版）。
浴室数据走的是更早的流程（真 COLMAP 估位姿 + StableNormal），见 `archive/HANDOFF_EnvGS_20261014.md` §5。

所有命令在 `~/SMBC/EnvGS/EnvGS` 下、`conda activate envgs` 里执行（第 2 步例外，见下）。

## 0. 输入：Blender 渲染的数据集

`~/SMBC/blender/project/proj_indoor_corridor_convex2/dataset_c2_480/`。生成流程见 blender 仓库 `README.md`，格式说明见数据集根目录的
`README.md`。预处理用到的部分：

| 路径 | 内容 | 谁用 |
|---|---|---|
| `persp/rgb/`、`persp/transforms.json` | RGB 图像、内参、GT 位姿 | 第 1 步 |
| `th_s02_format/extracted/pointclouds_multiecho/` | 每帧多回波 LiDAR（距离 + xyz，**不用回波类型**） | 第 2、4 步 |
| `th_s02_format/extracted/gt_poses_tum.txt` | LiDAR 传感器位姿 | 第 2、4 步 |
| `persp/depth/`、`persp/masks/`（镜面 = 3） | 真值 | **只用于评估**，以及 oracle 实验（第 6 步） |

相机 i 和 LiDAR 帧 i 是同一时刻（合成数据）。真机要换成时间最近的帧 + 位姿插值。

## 1. 稀疏点云和 EasyVolcap 格式（只用 RGB）

```bash
BDS=~/SMBC/blender/project/proj_indoor_corridor_convex2/dataset_c2_480; S=c2_480
O=data/datasets/original/corridor/$S; D=data/datasets/corridor
python mirror_lidar/tools/blender_gt_to_colmap.py $BDS $O --gpu 0
python scripts/preprocess/colmap_to_easyvolcap.py --data_root data/datasets/original/corridor --output $D --scenes $S --colmap sparse/0 --ext png
PYTHONPATH=$PWD python mirror_lidar/tools/compute_metadata_np126.py --data_root $D --scenes $S --eval
```

- `blender_gt_to_colmap.py`：**位姿不估计**，GT 位姿固定，COLMAP 只提特征、匹配、`point_triangulator` 三角化。坐标系就是 Blender 世界坐标（米，Z 朝上）。
  输出 `$O/{images → persp/rgb 软链, database.db, gt_model/, sparse/0/}`。c2_480：480/480 张，36.9 万点，重投影 0.82 px。
- `colmap_to_easyvolcap.py` → `$D/$S/{images, intri.yml, extri.yml, sparse/0/points3D.ply}`。
- `compute_metadata_np126.py`（官方 `compute_metadata.py` 的 numpy 1.26 兼容版）：打印 `spatial_scale` 和 `env_bounds`（点云 0.5%–99.5% 分位盒），
  **手动抄进数据集配置**。`--eval`：每 8 张取 1 张做验证（0, 8, 16, …），配置里的 `view_sample` 按这个写。
- 初始点云就是这个 COLMAP 点云（`sparse/0/points3D.ply`），里面有透过镜子三角化出的虚像点。LiDAR 初始化（`lidar_init_points.py`）试过，见 `TRAINING_RECIPE.md` §0。
- `envs/points3D.ply`（环境高斯初始化）不用准备：第一次训练时在 `env_bounds` 里随机撒点并存下来。

## 2. LiDAR 第一回波深度（`LidarDepthSupervisor` 用）

```bash
PYTHONPATH=~/SMBC/gsplat/gsplat/script ~/miniconda3/envs/gsplat/bin/python mirror_lidar/tools/lidar_first_echo_depth.py \
    $D/$S $BDS $D/${S}_lidar_depth --ratio 0.5 --eval
```

- 注意用 **gsplat 环境**的 python（要用那边的 LiDAR 读取代码）。
- 每帧每个 zone 取距离最近的回波，转到世界坐标，投到同编号相机（按训练分辨率 `--ratio 0.5` 缩放），存相机 z 深度。
- 输出 `$D/${S}_lidar_depth/<相机名>.npz`：`idx`（展平像素下标）、`z`、`H`、`W`。同一像素取最近的。
- `--eval` 对比 `persp/depth`：c2_480 中位误差 1.63 cm。
- 训练配置：`lidar_depth_dir`、`lidar_dpt_loss_weight 0.05`、`lidar_dpt_dead_zone 0.05`（残差 < 5 cm 不回传，约 2 倍测距噪声）。

## 3. 单目法线（StableNormal）

```bash
CUDA_VISIBLE_DEVICES=0 python submodules/StableNormal/run.py --data_root $D --scenes $S
```

- 输出 `$D/$S/normals/<相机名>/000000.png`，约 2 s/张，已存在的跳过。
- **镜子里预测的是镜中虚像的法线，是错的**，第 4 步会把镜面部分换成 LiDAR 拟合的法线。
- 中途被杀会留下截断的 PNG，加载时报 `'NoneType' object has no attribute 'ndim'`；训练前用 `cv2.imread(f) is None` 查一遍。

## 4. LiDAR 镜面拟合：法线先验 + 曲面（msurf 初始化）

```bash
F=data/result/envgs/corridor/mirror_surface_${S}_20261104
OMP_NUM_THREADS=8 python -u mirror_lidar/tools/lidar_mirror_normals_curved.py $D/$S $BDS --suffix lidarnorm_quadric \
    --surface quadric --rim-drop 0.06 --export-ply $F/quadric_rd06 --eval-sphere -0.04195 2.44195 2.23294 1.125
```

原理和参数选择详见 `MIRROR_SURFACE.md`。流程：
1. **找镜面**：每帧多回波 zone（7×7 开运算去噪）→ 连通块 → 光滑性检查（PCA 坐标系下 4 次高度场，MAD < 4 cm、内点 ≥ 70%）→ 3 cm 体素连通得到 patch。
2. **拟合**：patch 内的点在 1 cm (u, v) 网格上取均值（压噪声），`--surface quadric` 用一个全局隐式二次曲面（Taubin + Tukey IRLS），
   离边缘 `--rim-drop` 以内的格子才参与拟合（边缘点噪声大）。平面、球面都能精确表示。平面镜也可以用，或者用 `lidar_mirror_normals.py`（平面版）。
3. **范围（extent）**：1 cm 格子密度图 → 高斯模糊 → 大于内部中位数 1/4 → 闭运算 → 最大连通块 → 凸包。
   c2_480 已知问题：顶部约 22 cm 和右边缘最多 5.5 cm 没覆盖到（训练时可用 `extent_dilate` 外扩，见第 6 步）。
4. **每个相机**：相机光线和曲面求交，落在范围内且没有被同帧 LiDAR 遮挡的像素，法线换成拟合曲面法线；边缘腐蚀出的一圈（`--band`，默认 1 个 zone 间距）置黑（训练用 `norm_ignore_zero` 跳过）。

输出：
- `$D/${S}_lidarnorm_quadric/`：`images`、`intri/extri.yml`、`sparse` 都是指向 `$D/$S` 的软链，`normals/` 是替换后的法线。**训练用这个目录**。
- `$D/${S}_lidarnorm_quadric_debug/`：预览图和 `eval.json`。
- `$F/quadric_rd06/`：`patch0.npz`（坐标系、范围、高度场、二次曲面系数、格子均值，**可学镜面 msurf 的初始化**，配置里 `mirror_surface_cfg.fit`），
  `patch0_surface.ply` / `patch0_cells.ply`（MeshLab 看）、`patches.json`。
- `--eval-sphere`（GT 球心 + 半径，只评估）：c2_480 内部法线中位误差 0.128°，径向位置误差中位 −0.2 mm。GT 球面在 `proj_indoor_corridor_convex2/mirror_sphere.json`。

## 5. 镜面掩码（`MirrorRegionSupervisor` 用）

```bash
python mirror_lidar/tools/lidar_mirror_mask.py $D/${S}_lidarnorm_quadric $D/$S --eval-gt $BDS/persp
```

- 不重新跑 LiDAR：对比第 4 步的法线和原始 StableNormal 法线，被换掉的就是镜面。
- 输出 `$D/${S}_lidarnorm_quadric/mirror_masks/<相机名>/000000.png`（全分辨率 uint8）：255 镜面内部 / 128 边缘带 / 0 其它。
  不叫 `masks/`，否则 EasyVolcap 会当前景掩码读。
- c2_480 对 GT：内部精度 99.2%，召回 91.0%（右侧 74%：边缘带 13%、漏掉 13%，见 `mirror_lidar/tools/mask_lr_eval.py`），299/480 个相机有镜子。
- 训练配置：`mirror_mask_dir`。用到它的：`mirror_gs_norm`、msurf 深度/法线一致约束（只用 255）、镜面统计日志。

## 5b. 可视化检查：曲面投影、掩码叠加（视频）

```bash
V=data/result/envgs/corridor/prep_check_${S}_20261107
python mirror_lidar/tools/vis_mirror_prep.py surface $D/$S $F/quadric_rd06/patch0.npz $V/surface_init.mp4 --only-mirror --gpu 0
python mirror_lidar/tools/vis_mirror_prep.py mask $D/$S $D/${S}_lidarnorm_quadric/mirror_masks $V/mirror_masks.mp4 --only-mirror --gt $BDS/persp
```
两个视频都每 `--step` 个相机（默认 4）取一帧，`--only-mirror` 跳过没有镜子的帧；默认半分辨率、8 fps。已经写进 `prep_c2_480.sh` 的末尾。

- **`surface`：曲面初始化之后，把曲面投影回 RGB 图像**。
  - 用 msurf 自己的代码建曲面（`mirror_surface.py`，`--type spline`，和训练初始化完全一样），对每个像素的光线求交；
  - 命中且在范围内的像素按世界系法线着色，白线是曲面 (u, v) 每 10 cm 一条的网格线；
  - 黄线是范围边界；青线是 rim-drop 线，它外面那圈没参与拟合，是外推出来的；
  - `--extent-dilate 0.04` 按外扩后的范围画，原范围画成品红线。
  - **不做遮挡判断**：镜前的物体上也会画出曲面。
  - 要看的：曲面是否贴在镜面上，范围是否盖满镜面、有没有压到镜框，网格在镜缘有没有扭曲。
- **`mask`：生成的镜面掩码叠加到 RGB 上**。
  - 绿色是内部（255），黄色是边界带（128）；
  - `--gt <persp>` 时用品红线画 GT 镜面轮廓，帧上写这一帧的精度和召回。**GT 只用于评估**。
  - 也能看 oracle 掩码（`gt_mirror_masks.py` 的输出格式相同）。
- c2_480 的结果在 `data/result/envgs/corridor/prep_check_c2_480_20261107/`：
  - `surface_init.mp4`、`surface_init_dil4.mp4`（外扩 4 cm）；
  - `mirror_masks.mp4`（LiDAR 掩码 + GT 轮廓）、`mirror_masks_oracle.mp4`；
  - 每个 76 帧，约 36 MB。

## 6. 训练时才做的处理 / oracle 数据

- **范围外扩**：不改数据，配置 `mirror_surface_cfg.extent_dilate: 0.04`（米），加载 `patch0.npz` 时对范围做二值膨胀（日志 `extent dilated by 4 cm: 12004 -> 13492 cells`）。
- **GT 掩码（oracle，只用于对照实验）**：`python mirror_lidar/tools/gt_mirror_masks.py $BDS/persp $D/${S}_oracle_mirror_masks`，
  `persp/masks == 3` → 255，没有边缘带。配置里把 `mirror_mask_dir` 指过去，实验名带 `oracle`。
- 其它 oracle / 拟合曲面测试工具（`make_oracle_sphere_patch.py`、`export_oracle_sphere.py`、`export_fitted_quadric.py` 等）见各脚本开头的说明。

## 7. 数据集配置

`configs/datasets/corridor/c2_480_lidarnorm_quadric.yaml`：
- `data_root: data/datasets/corridor/c2_480_lidarnorm_quadric`；
- `view_sample`（训练 = 非 8 的倍数，验证 = 8 的倍数）；
- `preload_gs` / `env_preload_gs` 指向 `sparse/0/points3D.ply`、`envs/points3D.ply`；
- `spatial_scale`、`env_bounds` 抄第 1 步的输出。

实验配置（`configs/exps/envgs/corridor/envgs_c2_480_*.yaml`）再加上 `lidar_depth_dir`、`mirror_mask_dir`、`mirror_surface_cfg.fit` 等，
训练流程和各项 loss 见 `TRAINING_RECIPE.md`。

## 产物一览（c2_480）

| 路径 | 来源 | 内容 |
|---|---|---|
| `data/datasets/original/corridor/c2_480/` | 第 1 步 | COLMAP 工程 |
| `data/datasets/corridor/c2_480/` | 第 1、3 步 | EasyVolcap 格式 + StableNormal |
| `data/datasets/corridor/c2_480_lidar_depth/` | 第 2 步 | LiDAR 第一回波深度 |
| `data/datasets/corridor/c2_480_lidarnorm_quadric/` | 第 4、5 步 | 训练用：LiDAR 法线先验 + `mirror_masks/` + `envs/`（首次训练生成） |
| `data/datasets/corridor/c2_480_lidarnorm_quadric_debug/` | 第 4 步 | 预览、评估 |
| `data/result/envgs/corridor/mirror_surface_c2_480_20261104/` | 第 4、5 步 | 拟合曲面（`quadric_rd06/patch0.npz`）、日志 |
| `data/datasets/corridor/c2_480_oracle_mirror_masks/` | 第 6 步 | GT 掩码（oracle） |
| `data/result/envgs/corridor/prep_check_c2_480_20261107/` | 第 5b 步 | 检查视频：曲面投影、掩码叠加 |
