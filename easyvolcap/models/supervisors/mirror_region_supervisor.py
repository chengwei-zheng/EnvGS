# Mirror-region loss reweighting (2026-10-21, mirror_lidar/LIDAR_MIRROR_NORMAL.md).
# Per camera, <mirror_mask_dir>/<camera name>/000000.png: 255 = mirror interior, 128 = boundary band, 0 = rest
# (written by mirror_lidar/tools/lidar_mirror_mask.py from the LiDAR normal prior, full image resolution).
# Implemented as extra terms on top of VolumetricVideoSupervisor / EnvGSSupervisor, with the same per-pixel losses and
# the same normalization (mean over all pixels), so the effective per-pixel weights become:
#   normal prior: mirror interior norm_loss_weight + mirror_norm_loss_weight, elsewhere norm_loss_weight
#   L1 + SSIM:    mirror (interior + band) x mirror_rgb_scale before mirror_rgb_until_iter, x1 afterwards
#   depth-normal consistency (2026-10-27): mirror interior gs_norm_loss_weight (x depth scale) + mirror_gs_norm_loss_weight
#   (no depth scale), elsewhere gs_norm_loss_weight: against the small bumps on the mirror in the depth normals, which make
#   up only 1-2% of the full-image gs_norm loss (NEXT_STEPS 10-27)
#   blending weight beta (2026-10-30): mirror_beta_loss_weight * mean over the mirror interior of relu(mirror_beta_target - beta),
#   from mirror_beta_start_iter: pulls beta up on the mirror (on the c480 convex mirror beta stayed ~0.02 and the reflection was
#   baked into the surfel colours, NEXT_STEPS 10-30); normalized by the mirror pixel count, not the image
# msurf_* (2026-11-02): losses / stats of the explicit mirror surface (samplers/mirror_surface.py), summed into the loss.
# log_mirror_stats (2026-10-22): logging only, no loss: blending weight beta, reflection share (beta c_ref / RGB) and
# c_ref on the mirror interior of the training view and beta elsewhere (beta reset study).
import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.depth_utils import normalize_depth
from easyvolcap.utils.ssim_utils import ssim
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class MirrorRegionSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 mirror_mask_dir: str = '',
                 mirror_norm_loss_weight: float = 0.0,  # added on mirror interior pixels only
                 mirror_gs_norm_loss_weight: float = 0.0,  # 1 - n_render . n_depth on mirror interior pixels, mean over all pixels
                 mirror_gs_norm_start_iter: int = 0,
                 mirror_gs_norm_dist_max: float = 0.0,  # > 0: only on mirror interior pixels whose (detached) 2DGS distortion is below this
                 mirror_gs_norm_erode_px: int = 0,  # > 0: mirror gs_norm only on the interior eroded by this many px (full mask resolution)
                 mirror_rgb_scale: float = 1.0,  # mirror L1 / SSIM weight multiplier before mirror_rgb_until_iter
                 mirror_rgb_until_iter: int = 0,
                 img_loss_weight: float = 0.8,  # must match VolumetricVideoSupervisor (img_loss_type L1)
                 ssim_loss_weight: float = 0.2,
                 ssim_win_size: int = 11,
                 norm_ignore_zero: bool = False,
                 use_dpt_scale_norm_loss: bool = False,
                 log_mirror_stats: bool = False,
                 mirror_beta_loss_weight: float = 0.0,
                 mirror_beta_target: float = 0.9,
                 mirror_beta_start_iter: int = 3000,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.mirror_norm_loss_weight = mirror_norm_loss_weight
        self.mirror_gs_norm_loss_weight = mirror_gs_norm_loss_weight
        self.mirror_gs_norm_start_iter, self.mirror_gs_norm_dist_max = mirror_gs_norm_start_iter, mirror_gs_norm_dist_max
        self.mirror_gs_norm_erode_px = mirror_gs_norm_erode_px
        self.mirror_rgb_scale = mirror_rgb_scale
        self.mirror_rgb_until_iter = mirror_rgb_until_iter
        self.w_img, self.w_ssim, self.ssim_win_size = img_loss_weight, ssim_loss_weight, ssim_win_size
        self.norm_ignore_zero, self.use_dpt_scale_norm_loss = norm_ignore_zero, use_dpt_scale_norm_loss
        self.log_mirror_stats = log_mirror_stats
        self.mirror_beta_loss_weight, self.mirror_beta_target = mirror_beta_loss_weight, mirror_beta_target
        self.mirror_beta_start_iter = mirror_beta_start_iter
        self.masks, self.cache = [], {}
        if mirror_norm_loss_weight > 0 or mirror_gs_norm_loss_weight > 0 or (mirror_rgb_scale != 1 and mirror_rgb_until_iter > 0) or log_mirror_stats or mirror_beta_loss_weight > 0:
            names = sorted(os.listdir(mirror_mask_dir))  # camera_index i <-> i-th camera folder (sorted), as in the dataset
            self.masks = [cv2.imread(os.path.join(mirror_mask_dir, n, '000000.png'), cv2.IMREAD_UNCHANGED) for n in names]
            log(f'Mirror masks: {len(names)} cameras from {mirror_mask_dir}, '
                f'{sum((m == 255).any() for m in self.masks)} with mirror')
            if mirror_gs_norm_erode_px > 0:  # 255 -> 254 within erode_px of the interior border: excluded from the mirror gs_norm only
                k = np.ones((2 * mirror_gs_norm_erode_px + 1,) * 2, np.uint8)
                for m in self.masks:
                    m[(m == 255) & (cv2.erode((m == 255).astype(np.uint8), k) == 0)] = 254

    def mask(self, cam: int, H: int, W: int, device):
        k = (cam, H, W)
        if k not in self.cache:
            m = cv2.resize(self.masks[cam], (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1)
            self.cache[k] = torch.as_tensor(m, device=device)
        return self.cache[k]

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # explicit mirror surface (2026-11-02, samplers/mirror_surface.py): its LiDAR anchor / smoothness losses and logging
        if 'msurf_frac' in output: scalar_stats.msurf_frac = output.msurf_frac
        for k, v in output.get('msurf_losses', {}).items():
            scalar_stats[k] = v
            loss += v
        for k, v in output.get('msurf_stats', {}).items(): scalar_stats[k] = v
        if not self.masks: return loss
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        m = torch.stack([self.mask(c.item(), H, W, output.rgb_map.device) for c in batch.meta.camera_index])  # B, HW

        if self.log_mirror_stats: self.mirror_stats(output, m, scalar_stats)

        if self.mirror_norm_loss_weight > 0 and 'norm_map' in output and 'norm' in batch:
            # same per-pixel normal loss as EnvGSSupervisor
            norm_map = normalize(normalize(output.norm_map) @ batch.R.mT)
            norm = normalize(batch.norm * 2. - 1.)
            norm_loss = (norm_map - norm).abs().sum(dim=-1) + 1 - F.cosine_similarity(norm_map, norm, dim=-1)
            if self.norm_ignore_zero: norm_loss = norm_loss * (batch.norm.sum(dim=-1) > 1e-3).float()
            if self.use_dpt_scale_norm_loss: norm_loss = norm_loss * normalize_depth(output.dpt_map[..., 0].detach().clone())
            mirror_norm_loss = (norm_loss * (m >= 254).float()).mean()
            scalar_stats.mirror_norm_loss = mirror_norm_loss
            loss += self.mirror_norm_loss_weight * mirror_norm_loss

        if self.mirror_gs_norm_loss_weight > 0 and output.iter >= self.mirror_gs_norm_start_iter and 'norm_map' in output and 'surf_norm_map' in output:
            # same per-pixel term as EnvGSSupervisor's gs_norm_loss
            gs_norm = 1 - (output.norm_map * output.surf_norm_map).sum(dim=-1)  # B, HW
            w = (m == 255).float()  # with erode_px: the eroded interior (rim strip is 254)
            # dist_max (2026-10-27): with weight 1.0 on all interior pixels, the mirror-plane surfels grew 0-5 cm past the rim
            # (rim floaters, blurred wall behind the mirror). Presumably near the rim the mirror layer is not fully opaque, the
            # wall behind mixes into the expected depth, and extending opaque flat surfels outwards is the cheapest fix. The
            # accumulated alpha cannot flag this (closed room: ~1 everywhere), the distortion sum w_i w_j (m_i - m_j)^2 can:
            # 0.2 % of a wall 1 m behind ~1e-6, a 50/50 two-layer bump 5 cm apart ~6e-7, a single layer ~0. Detached mask.
            if self.mirror_gs_norm_dist_max > 0: w = w * (output.dist_map[..., 0].detach() < self.mirror_gs_norm_dist_max).float()
            scalar_stats.mirror_gs_norm_frac = w.sum() / (m >= 254).sum().clamp(min=1)
            mirror_gs_norm_loss = (gs_norm * w).mean()
            scalar_stats.mirror_gs_norm_loss = mirror_gs_norm_loss
            loss += self.mirror_gs_norm_loss_weight * mirror_gs_norm_loss

        if self.mirror_beta_loss_weight > 0 and output.iter >= self.mirror_beta_start_iter and 'spec_map' in output:
            w = (m >= 254).float()
            mirror_beta_loss = ((self.mirror_beta_target - output.spec_map[..., 0]).clamp(min=0) * w).sum() / w.sum().clamp(min=1)
            scalar_stats.mirror_beta_loss = mirror_beta_loss
            loss += self.mirror_beta_loss_weight * mirror_beta_loss

        if self.mirror_rgb_scale != 1 and output.iter < self.mirror_rgb_until_iter and 'rgb_map' in output:
            w = (m > 0).float()
            rgb_gt = batch.rgb + output.bg_color * (1 - batch.msk)  # as VolumetricVideoSupervisor.compute_image_loss
            l1_px = (output.rgb_map - rgb_gt).abs().mean(dim=-1)  # B, HW; L1 loss = mean over pixels of this
            to_img = lambda x: x.view(x.shape[0], H, W, 3).permute(0, 3, 1, 2)
            ssim_px = ssim(to_img(output.rgb_map), to_img(rgb_gt), data_range=1.0, win_size=self.ssim_win_size,
                           size_average=False).reshape(m.shape[0], -1)  # B, HW ssim map; SSIM loss = mean of 1 - this
            mirror_rgb_loss = self.w_img * (l1_px * w).mean() + self.w_ssim * ((1 - ssim_px) * w).mean()
            scalar_stats.mirror_rgb_loss = mirror_rgb_loss
            loss += (self.mirror_rgb_scale - 1) * mirror_rgb_loss  # net mirror weight = mirror_rgb_scale
        return loss

    @torch.no_grad()
    def mirror_stats(self, output: dotdict, m: torch.Tensor, scalar_stats: dotdict):
        if 'spec_map' not in output: return
        scalar_stats.ref_frac = output.ref_frac if 'ref_frac' in output else torch.ones((), device=m.device)  # traced pixel fraction
        beta = output.spec_map[..., 0]  # B, HW
        mi, other = m >= 254, m == 0
        if other.any(): scalar_stats.other_beta = beta[other].mean()
        if mi.sum() > 1000:
            scalar_stats.mirror_beta = beta[mi].mean()
            if 'ref_rgb_map' in output:  # beta * c_ref (no x2)
                ref = output.ref_rgb_map[mi].mean(-1)
                scalar_stats.mirror_ref_share = ref.sum() / output.rgb_map[mi].mean(-1).sum().clamp(min=1e-6)
                scalar_stats.mirror_cref = ref.sum() / beta[mi].sum().clamp(min=1e-6)
