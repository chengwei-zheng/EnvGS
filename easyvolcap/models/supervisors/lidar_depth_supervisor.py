# Sparse LiDAR depth supervision (2026-10-20, mirror_lidar/LIDAR_MIRROR_NORMAL.md).
# Per camera, <lidar_depth_dir>/<camera name>.npz holds the LiDAR samples projected into that camera at the training
# resolution: idx (flat pixel index, int64), z (camera z-depth, float32), H, W
# (written by mirror_lidar/tools/lidar_first_echo_depth.py). Loss: L1 between the rendered depth and z at those pixels.
# 2026-10-26: optional dead zone (lidar_dpt_dead_zone, ~2x the range noise sigma) so that the noise alone does not pull the
# surface around (bumps on the mirror with plain L1); a surface offset still gives a net pull from the samples beyond it.
# EasyVolcap's own depth path (dataset use_depths + DepthSupervisor) is not used: it needs EXR (this OpenCV build has no
# EXR codec) and its byte decoder slices [..., :3] on single-channel images.
import os
import numpy as np
import torch
from torch import nn

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class LidarDepthSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 lidar_depth_dir: str = '',
                 lidar_dpt_loss_weight: float = 0.0,
                 lidar_dpt_loss_start_iter: int = 0,
                 lidar_dpt_loss_weight_final: float = None,  # weight from lidar_dpt_loss_decay_iter on (None: keep)
                 lidar_dpt_loss_decay_iter: int = 0,
                 lidar_dpt_dead_zone: float = 0.0,  # (m) residuals below this give no loss / gradient: max(|pred - z| - dz, 0)
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.lidar_dpt_loss_weight = lidar_dpt_loss_weight
        self.lidar_dpt_loss_start_iter = lidar_dpt_loss_start_iter
        self.lidar_dpt_loss_weight_final = lidar_dpt_loss_weight_final
        self.lidar_dpt_loss_decay_iter = lidar_dpt_loss_decay_iter
        self.lidar_dpt_dead_zone = lidar_dpt_dead_zone
        self.samples = {}
        if lidar_dpt_loss_weight > 0:
            names = sorted(f for f in os.listdir(lidar_depth_dir) if f.endswith('.npz'))
            for i, f in enumerate(names):            # camera_index i <-> i-th camera folder (sorted), as in the dataset
                d = np.load(os.path.join(lidar_depth_dir, f))
                self.samples[i] = (torch.as_tensor(d['idx'], dtype=torch.long), torch.as_tensor(d['z'], dtype=torch.float),
                                   int(d['H']), int(d['W']))
            log(f'LiDAR depth: {len(names)} cameras from {lidar_depth_dir}, '
                f'{np.mean([len(v[0]) for v in self.samples.values()]):.0f} samples per camera')

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if self.lidar_dpt_loss_weight > 0 and 'dpt_map' in output and output.iter >= self.lidar_dpt_loss_start_iter:
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            losses = []
            for b in range(output.dpt_map.shape[0]):
                idx, z, h, w = self.samples[batch.meta.camera_index[b].item()]
                assert (h, w) == (H, W), f'LiDAR depth made for {h}x{w}, rendering {H}x{W}'
                pred = output.dpt_map[b, :, 0][idx.to(output.dpt_map.device)]
                losses.append(((pred - z.to(pred.device)).abs() - self.lidar_dpt_dead_zone).clamp_min(0).mean())
            lidar_dpt_loss = torch.stack(losses).mean()
            scalar_stats.lidar_dpt_loss = lidar_dpt_loss
            w = self.lidar_dpt_loss_weight
            if self.lidar_dpt_loss_weight_final is not None and output.iter >= self.lidar_dpt_loss_decay_iter:
                w = self.lidar_dpt_loss_weight_final
            loss += w * lidar_dpt_loss
        return loss
