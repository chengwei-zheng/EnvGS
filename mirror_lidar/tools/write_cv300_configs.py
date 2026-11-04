"""Dataset configs + env point cloud for scene B_convex (dataset_cv300, 2026-11-01; step 6/3 of data/logs/pipeline_cv300.sh).

Reads spatial_scale / env_bounds from the compute_metadata_np126.py log, writes configs/datasets/mirror_b/cv300_{lidarnorm,gtnorm}.yaml
(split as f300: every 8th frame = val) and data/datasets/mirror_b/cv300/envs/points3D.ply, generated once the way
EnvGSSampler.init_env_points does when the file is missing (seeded), so that both runs start from the same env points
(lidar_mirror_normals_curved.py / make_gtnorm_sphere.py copy it from cv300/).
    python mirror_lidar/tools/write_cv300_configs.py data/logs/metadata_cv300.log
"""
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from easyvolcap.utils.grid_utils import sample_points_subgrid  # noqa: E402
from easyvolcap.utils.colmap_utils import save_sfm_ply  # noqa: E402

N_FRAMES = 300
log = open(sys.argv[1]).read()
scale = float(re.search(r'spatial_scale: ([0-9.eE+-]+)', log).group(1))
bounds = [[float(v) for v in t.split(',')] for t in re.search(r'env_bounds: \[\[([^\]]+)\], \[([^\]]+)\]\]', log).groups()]
print('spatial_scale', scale, 'env_bounds', bounds)


def wrap(xs, n=24, indent=22):
    rows = [', '.join(str(x) for x in xs[i:i + n]) for i in range(0, len(xs), n)]
    return '[' + (',\n' + ' ' * indent).join(rows) + ']'


train = [i for i in range(N_FRAMES) if i % 8]
val = [i for i in range(N_FRAMES) if i % 8 == 0]
HEAD = {
    'lidarnorm': '# dataset_cv300 (scene B_convex: B_framed with the flat mirror replaced by an R 1 m convex cap, same trajectory / split as f300;\n'
                 '# 2026-11-01). normals/ = StableNormal + LiDAR-fitted curved mirror normal (lidar_mirror_normals_curved.py), boundary band black.',
    'gtnorm': '# dataset_cv300 as cv300_lidarnorm.yaml, but normals/ mirror pixels = GT sphere normal and mirror_masks = GT mirror pixels\n'
              '# (make_gtnorm_sphere.py, sphere from proj_indoor_mirror_b_convex/mirror_sphere.json; 2026-11-01).',
}
for k, head in HEAD.items():
    root = f'data/datasets/mirror_b/cv300_{k}'
    s = f"""configs: configs/datasets/mirror_b/mirror_b.yaml

{head} spatial_scale / env_bounds from compute_metadata_np126.py.

dataloader_cfg:
    dataset_cfg: &dataset_cfg
        data_root: {root}
        view_sample: {wrap(train)}

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        view_sample: {wrap(val)}

model_cfg:
    sampler_cfg:
        preload_gs: {root}/sparse/0/points3D.ply
        spatial_scale: {scale}
        # Environment Gaussian
        env_preload_gs: {root}/envs/points3D.ply
        env_bounds: {bounds}
"""
    p = f'configs/datasets/mirror_b/cv300_{k}.yaml'
    open(p, 'w').write(s)
    print('wrote', p)

ply = 'data/datasets/mirror_b/cv300/envs/points3D.ply'
if not os.path.exists(ply):
    torch.manual_seed(0)
    xyz = sample_points_subgrid(torch.as_tensor(bounds), 32, 5).float()
    rgb = torch.rand(xyz.shape, dtype=torch.float)
    os.makedirs(os.path.dirname(ply), exist_ok=True)
    save_sfm_ply(ply, xyz.numpy(), (rgb.numpy() * 255.0).astype(np.uint8))
    print('wrote', ply, xyz.shape[0], 'points')
