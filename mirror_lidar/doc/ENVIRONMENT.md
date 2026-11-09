# 环境、代码改动、评估、GUI、工具索引和坑

更新 2026-11-07，由 `archive/HANDOFF_EnvGS_20261014.md`（英文版，10-14）整理而来。
旧版里浴室数据的预处理命令和结果仍在那份文件里。

## 1. 目录

| 路径 | 内容 |
|---|---|
| `~/SMBC/EnvGS/EnvGS/` | EnvGS 仓库，基于上游 github.com/zju3dv/EnvGS 的 `0999347`（remote `origin`）；我们的改动提交在本地 main 上；fork 是 github.com/chengwei-zheng/EnvGS（remote `fork`）。**所有命令都在这里执行** |
| `EnvGS/data` → `~/SMBC/EnvGS/data/` | 软链（gitignore）：数据集、checkpoint、结果，见 `RESULTS.md` §1 |
| `EnvGS/data/logs/` | 启动、准备、评估脚本和日志（gitignore） |
| `EnvGS/mirror_lidar/doc/` | 文档（本目录） |
| `EnvGS/mirror_lidar/tools/` | 我们的脚本（旧路径 `~/SMBC/EnvGS/tools` 是指向它的软链） |
| `EnvGS/mirror_lidar/experiments/` | 10-17 到 10-28 的实验记录（json、配置、patch、图） |
| `EnvGS/patches/` | 子模块改动，用 `git -C submodules/<name> apply` 打上 |
| `EnvGS/build_variants/` | tracer 改 `START_OFFSET` 的副本（`dst_off005`、`dst_off020`），用法：`PYTHONPATH=build_variants/dst_off020/pkg` |
| `~/SMBC/gsplat/gsplat/script/eval_synthetic_glass.py` | GT 评估器（和 gsplat / RT-Splatting 共用，跑在 `gsplat` 环境里） |

硬件：4× RTX 3090。**GPU2 接显示器**（X11 `DISPLAY=:1`），只用来开 viewer，而且一次只开一个；训练用 GPU0/1/3。

## 2. 环境 `envgs`

`conda activate envgs`：Python 3.11，torch 2.3.1+cu118。编译 CUDA 代码一律用本地 CUDA 11.8：
```bash
export CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6
export CUDACXX=/usr/local/cuda-11.8/bin/nvcc   # diff-surfel-tracing 的 CMake 需要
```
- **安装要点**：
  - 所有包都用 `--no-build-isolation` 安装；
  - 版本固定：`pycolmap==0.6.1`、`imgui-bundle==1.6.2`、`setuptools<81`；diffusers 用 2024 年那一套（StableNormal 需要）；
  - 完整命令见 archive 版 §2。
- **检查 tracer 能用**：`diff_surfel_tracing` 包目录里必须有 `forward.ptx`、`backward.ptx`。
- 一些脚本要用其他环境：
  - `gsplat` 环境：LiDAR 读取、GT 评估器；
  - `colmap_cuda` 环境：COLMAP。
- 系统 python 没有 tensorboard，读 tensorboard 记录要用 `~/miniconda3/envs/envgs/bin/python`。

## 3. 相对上游的代码改动

| 文件 | 改动 |
|---|---|
| `easyvolcap/models/samplers/envgs_sampler.py` | 共用高斯反射 `shared_reflection`（默认 True，旧配置显式写 False）；bfilt（`specular_filtering_threshold` / `_pause_iter`）；β 软 reset `reset_specular_scale`；反射多采样 `ref_ss`；显式镜面 `mirror_surface_cfg`，输出 msurf 统计和目标 |
| `easyvolcap/models/samplers/mirror_surface.py`（新） | 显式镜面：`MirrorSurface`、`QuadricSurface`、`SplineSurface`，见 `MIRROR_SURFACE.md` |
| `easyvolcap/models/supervisors/lidar_depth_supervisor.py`（新） | LiDAR 稀疏深度 L1，带死区、权重衰减 |
| `easyvolcap/models/supervisors/mirror_region_supervisor.py`（新） | 镜面掩码相关的 loss 和日志：`mirror_gs_norm`、`mirror_norm`、`mirror_rgb_scale`、`mirror_beta`、msurf 锚定 / 正则 / 一致约束、`log_mirror_stats` |
| `easyvolcap/models/supervisors/envgs_supervisor.py` | `norm_ignore_zero`：先验法线为黑色的像素不算法线 loss |
| `easyvolcap/utils/gaussian2d_utils.py` | `reset_specular` 支持按比例缩放（`reset_specular_scale`） |
| `easyvolcap/utils/optix_utils.py` | 共用版反射传 `override_color` 时，上游访问了没定义的 `pcd.feature_splatting`，改成 `getattr` |
| `submodules/diff-surfel-tracing` | OptiX `validationMode` 改成 OFF，快 15–25%，结果不变（`patches/diff-surfel-tracing_validation_off.patch`） |
| `configs/datasets/{bathroom,mirror_a,mirror_b,corridor}/`、`configs/exps/envgs/{...}/`、`configs/specs/*_viewer.yaml` | 数据集、实验、viewer 配置 |

## 4. 训练

```bash
r=c2_480_msurf_spline_learn_ldepth005_dz5_scons
echo "evc-train -c configs/exps/envgs/corridor/envgs_$r.yaml exp_name=envgs/corridor/envgs_$r" > data/logs/cmd_envgs_corridor_$r.txt
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $(cat data/logs/cmd_envgs_corridor_$r.txt) < /dev/null > data/logs/envgs_corridor_${r}_train.log 2>&1
```
- 实际都写成 `data/logs/launch_*.sh`：训练后自动评估和分析，用 `setsid` 启动。可以照 `launch_scons_dil4_oraclemask_1107.sh` 写。
- `exp_name` 决定 `trained_model/`、`record/`、`result/` 下的目录。
- 进度用 tensorboard 的 step 判断。进度条不进日志；日志里的 epoch 只在存 checkpoint 时打印。

## 5. 评估

每个场景一个脚本：`data/logs/eval_<场景>.sh <run> <gpu>`，比如 `eval_c2_480.sh`、`eval_cv300.sh`、`eval_f300_lidar.sh`、`eval_r300lp_lidar.sh`。流程：
1. `tools/export_raw.py`：EasyVolcap `-t test`，RENDER / DEPTH 存成 float `.npy`（envgs 环境的 OpenCV 5.0 写不了 EXR）→ `result/.../envgs_<run>/rawdepth/`。
2. `tools/envgs_to_eval_npz.py` → `eval_npz/`。
3. `tools/eval_glass_anyname.py`（包装 gsplat 的 `eval_synthetic_glass.py`，`--downscale 2` 即 1296×972）：
   - 输出全图、不透明区、镜面的 PSNR，以及深度对 GT 的误差；
   - 结果在 `eval_gt/summary_val.json`，日志 `eval_gt.log`。
4. 镜面专项：
   - `tools/mirror_perc_b.py`：镜面 SSIM / LPIPS-a / PSNR，**镜面比较以它为准**；
   - `geom_eval_b.py`：几何对 GT 网格；
   - `mirror_bumps_b.py`：镜面鼓包；
   - `mirror_normal_sphere.py`：曲面镜法线对 GT 球面；
   - 环境变量 `RESULT_GROUP`、`GT_PERSP`、`CEILING`、`MIRROR_SPHERE` 指定场景。
5. 走廊的分段统计（不在 eval 脚本里，launch 脚本里单独调用）：`mirror_lr_bins.py`、`mirror_lr_depth.py`、`mirror_lr_depth_extent.py`、`spline_normal_err.py`、`mask_lr_eval.py`。

注意：
- **镜面比较看 LPIPS / SSIM 为主**：共用版镜中画面清晰，但有小的位置偏差，PSNR 会吃亏。
- 训练末段镜面指标有约 ±0.8 dB 的波动。

## 6. GUI（`evc-gui`）

用 `mirror_lidar/tools/gui.sh <run>`（11-09 起；`-l` 列出 run，`-n` 只打印）。它自动找组和 viewer 配置，run 名带 `_off0NN` 时自动用 `build_variants/dst_off0NN` 的光追（和训练一致）；默认 GPU2（`GPU=` 可改），已有 viewer 在开时拒绝再开（`FORCE=1` 强制）。旧的 `gui_*.sh` 清单见 `RESULTS.md` §6。关窗口时不保存。
- **`exp_name` 必须和训练时一样**，否则会悄悄加载一个没训练的模型。
- viewer 配置在 `configs/specs/{mirror_a,mirror_b,corridor,corridor_c2,bathroom}_viewer.yaml`：初始相机用一个训练相机的 R/T，`focal_ratio`，`imbound_crop: False`，world_up。
  GUI 模式下验证集被换成 `NoopDataset`，不给这些参数的话，相机会在房间外面，视场也不对。
- 操作：
  - W/S 前后，A/D 左右，V/C 上下，Q/E 旋转；
  - 左键拖动绕点转，右键拖动转视角，Shift+左键或中键拖动平移；
  - Ctrl+Shift+滚轮沿视线移动，滚轮改焦距，Alt+滚轮改渲染分辨率。

## 7. 工具索引（`mirror_lidar/tools/`）

每个脚本开头都有用法说明。

| 类别 | 脚本 |
|---|---|
| 数据准备 | `blender_gt_to_colmap.py`、`compute_metadata_np126.py`、`lidar_first_echo_depth.py`、`lidar_mirror_normals.py`（平面）、`lidar_mirror_normals_curved.py`（曲面）、`quadric_mls.py`、`lidar_mirror_mask.py`、`lidar_init_points.py`、`lidar_plane_depth.py`、`write_cv300_configs.py`、`vis_mirror_prep.py`（预处理检查视频：曲面投影、掩码叠加） |
| oracle / 消融数据 | `make_mirror_normal_variants.py`（gtnorm / nmask）、`make_gtnorm_sphere.py`、`make_oracle_sphere_patch.py`、`gt_mirror_masks.py` |
| 评估 | `export_raw.py`、`envgs_to_eval_npz.py`、`eval_glass_anyname.py`、`eval_{bathroom,mirror_a,mirror_b}_run.sh`、`per_mirror.py`、`per_obj_mirror_b.py`、`mirror_perc_b.py`、`geom_eval_b.py`、`mirror_bumps_b.py`、`mirror_flatness_b.py`、`mirror_normal_spread_b.py`、`mirror_normal_sphere.py`、`mirror_prior_vs_gt_b.py` |
| 镜面分段诊断 | `mirror_rim_band.py`（按离镜缘距离）、`mirror_lr_bins.py`（按水平位置 / 反射距离）、`mirror_lr_depth.py`、`mirror_lr_depth_extent.py`、`mirror_lr_lidar.py`、`mask_lr_eval.py`、`spline_normal_err.py`、`frame_hit_stats.py`（镜框与 START_OFFSET） |
| 推理对照（替换反射几何） | `export_oracle_sphere.py`（GT）、`export_fitted_quadric.py`、`export_fitted_nmap.py`（LiDAR 拟合）、`downsample_raw.py`（超采样） |
| 分析 / 检查 | `mirror_normal_grads.py`（各 loss 对镜面法线的梯度）、`check_dist_loss.py`（distortion 实现）、`prune_mirror_front.py`（删镜前面元）、`quadric_mls_synth.py`（拟合合成测试） |
| viewer | `gui_*.sh`（见 `RESULTS.md` §6） |

## 8. 坑

1. **diff-surfel-tracing 没有 PTX 也会"装成功"**：CMake 找到了系统的 CUDA 10.1。
   - 症状：从第 3000 步起刷 `Optix Error: 'Invalid value'`；
   - 解决：设 `CUDACXX`，删掉 `build/`，重装。
2. **EasyVolcap 出异常后进 pdb 卡住**：一律用 `< /dev/null` 运行。
3. **EasyVolcap 在 import 时就解析 `sys.argv`**：包装脚本要先设好 `sys.argv` 再 import。
4. **长任务用 `setsid … &` 启动**：从工具 shell 里用 `nohup` 起的任务，会随 shell 一起被杀。
5. **不要在自己的命令行里 `pkill -f <pattern>`**：pattern 会匹配到当前 shell 自己，把它杀掉。要按 PID 杀。
6. **StableNormal 被中途杀掉会留下截断的 PNG**：加载时报 `'NoneType' object has no attribute 'ndim'`，训练前要检查。
7. **改 tracer 的 `START_OFFSET`** 要另编一份 ptx（`build_variants/`），启动日志第一行会打印实际加载的 tracer 路径。
   推理时改小 offset 会打到镜面上方那层高斯，所以要验证 offset 的影响只能重训。
8. **OpenCV 5.0 没有 EXR writer**，EasyVolcap 的 `save_image` 写 `.exr` 会静默失败（所以有 `export_raw.py`）。
9. scratchpad 里自己写的脚本名不要和标准库重名（比如 `inspect.py`），否则会顶掉标准库。
