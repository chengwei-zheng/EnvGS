# EnvGS on the synthetic bathroom datasets — handoff notes

State as of 2026-10-14. **Newer status (2026-10-29): the mirror work continues on scene B_framed (`mirror_b/f300`); its default setting is
`envgs_f300_lidarnorm_shared_bfilt_mgsn05`. Current results / recipe: `mirror_lidar/TRAINING_RECIPE.md` §3, `mirror_lidar/NEXT_STEPS.md`.** Everything below was set up and run from a separate Claude Code session (working in
`~/SMBC/gsplat/gsplat`); this file is the single reference for continuing the EnvGS work.

Contents: 1 Layout · 2 Environment `envgs` · 3 Local changes to the EnvGS repo · 4 Official data (audi) ·
5 Custom data pipeline (bathroom) · 6 GT evaluation · 7 Experiments and results · 8 GUI · 9 Gotchas ·
10 Suggested next steps

---

## 1. Layout

| Path | What |
|---|---|
| `~/SMBC/EnvGS/EnvGS/` | EnvGS repo, based on github.com/zju3dv/EnvGS commit `0999347` (remote `origin`); our changes are committed on top; fork: github.com/chengwei-zheng/EnvGS (remote `fork`). EasyVolcap-based. Run every command from here. |
| `~/SMBC/EnvGS/EnvGS/data` | **symlink → `../data`** (gitignored). All datasets, records, results, checkpoints live in `~/SMBC/EnvGS/data/`. |
| `~/SMBC/EnvGS/data/datasets/` | `envgs/{audi,dog}`, `original/envgs/{audi,dog}` (COLMAP format), `refnerf/{ref_real,shiny_blender}`, `mmw421_samsung` (926 views, not referenced by any config, untouched), `bathroom/…` (ours, §5), `original/bathroom/…` (symlinks, §5). Original zips were deleted after extraction (per user). |
| `~/SMBC/EnvGS/data/record/<exp_name>/` | tensorboard + resolved config yaml of each run |
| `~/SMBC/EnvGS/data/trained_model/<exp_name>/` | checkpoints (`latest.npz` / `latest.pt`) |
| `~/SMBC/EnvGS/data/result/<exp_name>/` | `metrics.json` + RENDER/DEPTH/NORMAL/SPECULAR/DIFFUSE/REFLECTION pngs of the test views (written at the end of training) |
| `~/SMBC/EnvGS/EnvGS/data/logs/` | `cmd_envgs_<run>.txt` (exact train command), `envgs_<run>_train.log`, export / GUI / StableNormal logs |
| `~/SMBC/EnvGS/EnvGS/mirror_lidar/tools/` | our helper scripts (export, conversion, GT eval, GUI launcher); in the repo since 2026-10-18 (old path `~/SMBC/EnvGS/tools` is a symlink) |
| `~/SMBC/EnvGS/EnvGS/mirror_lidar/experiments/` | experiment records (json + exp/resolved yaml + patch + figures), one prefix per experiment (old path `~/SMBC/EnvGS/experiments` is a symlink) |
| `~/SMBC/EnvGS/EnvGS/patches/` | local changes to submodules (e.g. `diff-surfel-tracing` validationMode OFF), apply with `git -C submodules/<name> apply` |
| `~/SMBC/gsplat/gsplat/script/eval_synthetic_glass.py` | the GT evaluator shared with the gsplat / RT-Splatting comparisons (runs in the `gsplat` conda env) |

Hardware notes: 4× RTX 3090. **GPU2 drives the display** (X11, `DISPLAY=:1`, `XAUTHORITY=/run/user/1000/gdm/Xauthority`);
GPU0/1/3 are free for training.

---

## 2. Environment `envgs`

`conda activate envgs` — Python 3.11, torch 2.3.1+cu118. Always build CUDA code with the local CUDA 11.8:

```bash
export CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6
export CUDACXX=/usr/local/cuda-11.8/bin/nvcc   # needed for diff-surfel-tracing's CMake step, see below
```

How it was built (README recipe + fixes):

```bash
conda create -n envgs "python=3.11" -y && conda activate envgs
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# requirements.txt / requirements-dev.txt installed one package at a time (README's sed|awk|xargs loop) with
# --no-build-isolation, skipping bare torch/torchvision/torchaudio so the cu118 build is not replaced
pip install -e . --no-build-isolation --no-deps
git submodule update --init --recursive
pip install -v --no-build-isolation submodules/diff-surfel-tracing          # needs CUDACXX, see gotcha 1
pip install --no-build-isolation submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet \
    submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch05 \
    submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch07
# fixes for packages the README loop mangles / that are too new:
pip install "setuptools<81" scikit-build-core pybind11
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip install "git+https://github.com/wookayin/gpustat" "cuda-python<12.9" regex
pip install "pycolmap==0.6.1" "imgui-bundle==1.6.2"
# StableNormal (monocular normals for custom data): sh submodules/StableNormal/setup.sh, then pin the 2024 stack
pip install "diffusers==0.29.2" "transformers==4.44.2" "huggingface_hub==0.24.6" "accelerate==0.33.0"
```

Key versions now installed: torch 2.3.1+cu118, numpy 1.26.4, pycolmap 0.6.1, imgui-bundle 1.6.2,
opencv-python 5.0.0 (no EXR writer!), xformers 0.0.27, diffusers 0.29.2, transformers 4.44.2, tinycudann, pytorch3d,
detectron2, simple_knn, nvdiffrast, torch_scatter. `torchmcubes` failed to build (CMake) — only used for mesh export, ignored.

Verify the ray tracer is really usable (its PTX kernels must be present):
```bash
ls $(python -c "import diff_surfel_tracing,os;print(os.path.dirname(diff_surfel_tracing.__file__))")   # must list forward.ptx backward.ptx
```

---

## 3. Local changes to the EnvGS repo (`git status`)

| Change | Why |
|---|---|
| `submodules/diff-surfel-tracing/optix_tracer/optix_wrapper.cpp`: `validationMode = …_ALL` → `…_OFF` | OptiX debug validation; OFF is ~15–25 % faster, results unchanged. Revert with `git checkout` in the submodule + reinstall. |
| `easyvolcap/models/supervisors/envgs_supervisor.py`: new option `norm_ignore_zero` (default `False`) | When `True`, pixels whose prior normal image is pure black (0,0,0) are excluded from the monocular normal loss. Used only by the `_nmask` ablation (§7). Default keeps original behaviour. |
| new `configs/datasets/bathroom/{bathroom,walk400v2,walk400_2mirrors,walk400_2mirrors_nmask}.yaml` | bathroom dataset configs (§5) |
| new `configs/exps/envgs/bathroom/envgs_walk400{v2,_2mirrors,_2mirrors_nmask,_2mirrors_spec001,_2mirrors_spec01}.yaml` | experiment configs (§7) |
| new `configs/specs/bathroom_viewer.yaml` | GUI defaults for the bathroom scenes (§8) |
| `data` → `../data` symlink | gitignored |

---

## 4. Official data (audi) — sanity check

```bash
cd ~/SMBC/EnvGS/EnvGS && conda activate envgs
CUDA_VISIBLE_DEVICES=0 evc-train -c configs/exps/envgs/envgs/envgs_audi.yaml exp_name=envgs/envgs/envgs_audi < /dev/null
```
60k steps (120 epochs × 500), 4.3 h on a 3090. Test (26 views): **PSNR 28.85 / SSIM 0.882 / LPIPS 0.139**
vs paper Table 3: 28.63 / 0.876 / 0.121 → reproduced. `dog` was not run.

Speed profile (read `TRAIN/batch` from tensorboard; the progress bar is not in the log): 0.04 s/step before step 3000,
~1.7 s/step right after (ray tracing starts at `render_reflection_start_iter=3000` together with an opacity reset to
0.01 → many traversals), then 0.17–0.3 s/step.

---

## 5. Custom data pipeline (bathroom)

Source data: `~/SMBC/blender/project/proj_indoor_bathroom/{dataset_walk400, dataset_walk400_2mirrors}`
(`dataset_walk400` = the 2026-10-07 re-render, called **walk400v2** here). Poses come from COLMAP on
`th_s02_format/extracted/{images,sparse/0}` (400/400 registered, ATE vs GT < 0.5 mm; `_2mirrors` reuses the
walk400v2 COLMAP model — identical cameras). Ground truth for evaluation is in each dataset's `persp/`.

```bash
cd ~/SMBC/EnvGS/EnvGS && conda activate envgs
B=~/SMBC/blender/project/proj_indoor_bathroom; O=data/datasets/original/bathroom
# 1) COLMAP-format view (symlinks only)
for s in walk400v2:dataset_walk400 walk400_2mirrors:dataset_walk400_2mirrors; do n=${s%%:*}; d=${s##*:}
  mkdir -p $O/$n/sparse; ln -sfn $B/$d/th_s02_format/extracted/images $O/$n/images
  ln -sfn $B/$d/th_s02_format/extracted/sparse/0 $O/$n/sparse/0; done
# 2) COLMAP -> EasyVolcap (images are png!)
python scripts/preprocess/colmap_to_easyvolcap.py --data_root data/datasets/original/bathroom \
    --output data/datasets/bathroom --scenes walk400v2 walk400_2mirrors --colmap sparse/0 --ext png
# 3) metadata: test views = every 8th (frames 0,8,…,392 — same split as gsplat/RT-Splatting), spatial_scale, env_bounds
python scripts/preprocess/tools/compute_metadata.py --data_root data/datasets/bathroom --scenes walk400v2 walk400_2mirrors --eval
# 4) monocular normals (StableNormal, ~2 s/img, writes normals/<cam>/000000.png; skips existing files)
CUDA_VISIBLE_DEVICES=1 python submodules/StableNormal/run.py --data_root data/datasets/bathroom --scenes walk400v2 < /dev/null
CUDA_VISIBLE_DEVICES=3 python submodules/StableNormal/run.py --data_root data/datasets/bathroom --scenes walk400_2mirrors < /dev/null
# 5) configs (already written): dataset ratio 0.5 = 1296x972 training/eval resolution, schedule copied from envgs_audi
#    (120 epochs, perc_loss_start_iter 45000). Both scenes share spatial_scale 7.7118 and env_bounds.
# 6) train (detached, see gotcha 6)
s=walk400_2mirrors; echo "evc-train -c configs/exps/envgs/bathroom/envgs_$s.yaml exp_name=envgs/bathroom/envgs_$s" > data/logs/cmd_envgs_$s.txt
setsid bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs; CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 exec \$(cat data/logs/cmd_envgs_$s.txt) < /dev/null > data/logs/envgs_${s}_train.log 2>&1" &
```

Notes: `envs/points3D.ply` (environment-Gaussian init) does not exist for custom data — EnvGS samples random points
inside `env_bounds` on the first run and saves them there (all `_2mirrors*` runs share that file). Training ~5.5–9 h
per scene on a 3090. **StableNormal is wrong inside the large (round, above-sink) mirror**: it predicts the normals of
the reflected room; it is fine on the oval wall mirror. Check normals before training: a StableNormal run that was
killed mid-write left a truncated PNG (crash `'NoneType' object has no attribute 'ndim'` at data loading) —
detect with `cv2.imread(f) is None` over `normals/*/*.png`.

Ablation data `data/datasets/bathroom/walk400_2mirrors_nmask/`: images/sparse/intri/extri symlinked to
`walk400_2mirrors`, normals copied with GT glass+mirror pixels (`persp/masks` ∈ {2,3}) set to black.

---

## 6. GT evaluation

EasyVolcap's own test metrics land in `data/result/<exp>/metrics.json`. For per-region GT evaluation (opaque /
glass / mirror PSNR, depth vs GT, which GT layer the depth matches) we export raw RGB + depth and reuse the
evaluator used for gsplat and RT-Splatting:

```bash
~/SMBC/EnvGS/EnvGS/mirror_lidar/tools/eval_bathroom_run.sh <run_tag> <dataset_dir_name> <gpu>
# e.g. eval_bathroom_run.sh walk400_2mirrors_spec01 dataset_walk400_2mirrors 0
```
which does:
1. `tools/export_raw.py` — EasyVolcap `-t test` with linear depth, but RENDER/DEPTH are saved as float `.npy`
   (OpenCV 5.0 in `envgs` has no EXR writer; EasyVolcap's `save_image` fails silently on `.exr`).
   Output `data/result/<exp>/rawdepth/{RENDER,DEPTH}/frame0000_cameraXXXX.npy`. EnvGS depth is **z-depth**
   (`use_z_depth=True`) in COLMAP units.
2. `tools/envgs_to_eval_npz.py` — per-frame npz named by image stem (camera k = k-th sorted image).
3. `eval_synthetic_glass.py --pred_dir … --pred_depth_key depth --downscale 2` (gsplat env; Sim(3)-aligns COLMAP to
   GT poses for the metric depth scale) → `data/result/<exp>/eval_gt/summary_val.json`, `per_frame_val.json`, `vis/`.
4. `tools/per_mirror.py` — splits the mirror stats by object (round `mirror` vs `WallMirror`) via `persp/obj_id`.

---

## 7. Experiments and results

All at 1296×972, 50 test views, same GT evaluator. RT-Splatting rows are from `~/SMBC/RT-Splatting` (same split).

| Run (`envgs_*`) | Data | Change | PSNR | Mirror PSNR | Round mirror: depth = virtual / = plane | Wall mirror: PSNR, virtual / plane |
|---|---|---|---|---|---|---|
| `walk400v2` | 1 mirror | — | 35.00 | 35.4 | 79 % / 2 % | — |
| `walk400_2mirrors` | 2 mirrors | — | 34.99 | 32.5 | 82 % / 2 % | 28.3, 48 % / 24 % |
| `walk400_2mirrors_nmask` | 2 mirrors | no mono-normal loss on GT mirror+glass | 34.89 | 32.5 | 75 % / 3 % | 28.2, 51 % / 12 % |
| `walk400_2mirrors_spec001` | 2 mirrors | `init_specular=0.01` (default 1e-3) | 34.89 | 32.4 | 78 % / 2 % | 28.3, 48 % / 28 % |
| `walk400_2mirrors_spec01` | 2 mirrors | `init_specular=0.1` | 34.47 | 31.9 | 78 % / 2 % | 27.9, 49 % / 27 % |
| RT-Splatting walk400v2 | 1 mirror | — | 36.22 | 35.9 | 74 % / 2 % | — |
| RT-Splatting 2mirrors | 2 mirrors | — | 36.06 | 32.9 | 71 % / 2 % | 29.6, 45 % / 26 % |

Other findings: EnvGS has the most accurate opaque geometry (median depth error 0.3 cm vs 0.5 cm RT-Splatting), but
does not reconstruct the shower glass surface (RT-Splatting does). ("virtual" = rendered depth within 10 % of
`gt_layers/depth_reflected`.)

**Conclusion:** none of the methods reconstructs the planar mirrors. EnvGS's DIFFUSE layer carries the mirror images
(a mirrored "virtual room" behind the glass plane); its specular/reflection branch stays ~unused. Removing the wrong
normal prior or raising `init_specular` does not change that (0.1 only adds global specular noise on walls).
Interpretation: for a planar mirror whose back is never observed, "virtual room behind a window" explains every view
exactly, so RGB-only optimisation has no reason to prefer "mirror plane + traced reflection". Ray tracing is the right
forward model but needs a geometric prior to break the ambiguity (LiDAR glass/mirror-surface returns, mirror mask +
planarity). The oval wall mirror behaves differently (shallow bulge, blurry, PSNR −6 dB) — it appears to be mounted on
the shower glass so its back is visible from some views, which forbids a full virtual room; to be confirmed with the
scene file.

---

## 8. GUI (`evc-gui`)

Bathroom runs (adds `configs/specs/bathroom_viewer.yaml`: `world_up [-0.045,-0.994,-0.099]` because this COLMAP world
is not Z-up — with the default `[0,0,1]` mouse drags roll —, orbit origin = mean camera center, `movement_force: 10`):
```bash
~/SMBC/EnvGS/EnvGS/mirror_lidar/tools/gui_bathroom.sh walk400_2mirrors          # GPU2 by default; GPU=x to override
```
Official scenes: `CUDA_VISIBLE_DEVICES=2 evc-gui -c configs/exps/envgs/envgs/envgs_audi.yaml exp_name=envgs/envgs/envgs_audi viewer_cfg.window_size=540,960`.
**Always pass the same `exp_name` as training**, otherwise it silently loads an untrained model.

Controls: W/S forward/back (horizontal), A/D left/right, V/C up/down, Q/E and 1/2 rotate; left-drag orbit,
right-drag look around, Shift+left or middle drag pan; **Ctrl+Shift+scroll = move along the view direction**;
plain scroll changes focal length; Alt+scroll render resolution. Panel: "World up", "Mouse speed".

---

## 9. Gotchas

1. **diff-surfel-tracing installs "successfully" without its PTX kernels** if CMake picks the system `/usr/bin/nvcc`
   (CUDA 10.1): `setup.py` runs `cmake && make` via `os.system` and ignores failure. Symptom: from step 3000 on,
   endless `Optix Error: 'Invalid value'` at `trace_surfels.cpp:306/463`. Fix: `export CUDACXX=/usr/local/cuda-11.8/bin/nvcc`,
   `rm -rf submodules/diff-surfel-tracing/build`, reinstall with `--no-build-isolation`, check the `.ptx` files.
2. Newer pip builds in isolation → CUDA extensions can't see torch: use `--no-build-isolation`.
3. pycolmap 4.x breaks `easyvolcap/utils/undist_utils.py` → `pycolmap==0.6.1`. imgui-bundle 1.92 breaks the GUI
   (`push_font`) → `1.6.2`. Latest diffusers needs torch ≥ 2.5 (`torch.accelerator`) → diffusers stack pinned (§2).
4. On an exception EasyVolcap drops into a pdb post-mortem and **hangs**; always run with `< /dev/null`.
5. EasyVolcap parses `sys.argv` at import time — wrapper scripts must set `sys.argv` before importing easyvolcap.
6. Launch long jobs with `setsid … &`. A job started with plain `nohup … &` from inside a tool/background shell
   was killed by SIGTERM when that wrapper shell was stopped.
7. Don't `pkill -f <pattern>` from a shell whose own command line contains the pattern (kills the shell itself);
   kill by PID (`data/logs/*.pid`).
8. Training/eval resolution `ratio: 0.5`; the GT evaluator compares at `--downscale 2` (1296×972).

---

## 10. Suggested next steps (discussed with the user)

Move to simpler controlled scenes before adding priors: no shower glass, one mirror per scene —
(A) wall-mounted mirror whose back is never visible (pure ambiguity case: baseline must fail, any prior's effect is
clear); (B) free-standing / partition mirror whose back is visible from some views (ambiguity removed by geometry: tests
whether the reflection itself is modelled). Trajectory: objects reflected in the mirror should also be seen directly
from many views; include oblique / grazing views and the mirror frame; include near-frontal views so the simulated
LiDAR returns `glass_surface` echoes from the mirror. ~150–200 frames is enough for fast iteration. Then try a mirror
prior: LiDAR mirror-surface depth or mask + planarity constraint.
