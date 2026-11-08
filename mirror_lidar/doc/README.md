# mirror_lidar 文档

EnvGS + 多回波 LiDAR 做镜面场景重建（共用一套高斯做反射、LiDAR 拟合的显式镜面）。总计划见 `~/SMBC/blender/paper_2610/shared_gaussians_lidar_plan.md`。

**接手先读**：`NEXT_STEPS.md`（现状和在跑的实验）→ `RESULTS.md`（数据和结果）→ `TRAINING_RECIPE.md`（当前方法）。

| 文档 | 内容 |
|---|---|
| `NEXT_STEPS.md` | 现状、在跑的实验、待做 |
| `RESULTS.md` | 场景清单、已生成数据的位置、各阶段主要结论和结果表、viewer 脚本 |
| `TRAINING_RECIPE.md` | 当前主线配置；每次迭代的渲染步骤、训练中的特殊操作、全部 loss；已知问题 |
| `MIRROR_SURFACE.md` | 多回波 LiDAR 镜面拟合（检测、二次曲面、范围、掩码）；显式镜面 msurf 和高斯与曲面一致约束 |
| `DATA_PREP.md` | 数据预处理：Blender 数据 → EnvGS 训练数据，每一步的命令、输入输出 |
| `ENVIRONMENT.md` | 目录、conda 环境、相对上游的代码改动、训练和评估流程、GUI、工具索引、坑 |
| `archive/` | 旧版文档原样保留，不再更新：10-22 到 11-05 的完整实验日志、10-25 版训练流程、10-14 英文交接文档、11-02 版 LiDAR 镜面文档 |

另外：`../experiments/` 是 10-17 到 10-28 的实验记录（json + 配置 + 图）。

## 约定
- **命名**：
  - 用了 GT 的对照叫 `oracle`（实验名、脚本名、环境变量里都带）；
  - 只用 LiDAR 拟合的叫 `fitted`；
  - 实验名后缀含义：`lidarnorm`（LiDAR 法线先验）、`shared`（共用高斯）、`bfilt`、`mgsnXX`（镜面 gs_norm 权重）、`msurf_spline_learn`（可学 B 样条镜面）、
    `ldepth005_dz5`（LiDAR 深度 0.05、5 cm 死区）、`scons`（高斯与曲面一致约束）、`dilN`（范围外扩 N cm）。
- **镜面方位**：走廊镜子的 u < 0 是 viewer 图像的右侧。
- 渲染和评估结果**不覆盖**，新结果放新目录（推理对照写 `envgs_<run>_<tag>/`）。
- 所有 EnvGS 实验都用共用高斯（`shared_reflection: True`）。
- 只在本地 commit，用户要求时才 push。
