"""
Loss functions for Card/ID ROI Detection

Loss structure
--------------
Total = w_pixel  × VectorizedOHEM (+ Focal Weighting)
      + w_dice   × Dice
      + w_iou    × IoU
      + w_boundary × DT_Boundary
      + 0.5      × Sobel_Boundary
      + w_class  × CrossEntropy

Design rationale
----------------
- VectorizedOHEM : removes B×C for-loop, focuses on hard pixels
- Focal Weighting: auto-weights difficult pixels (boundary, small cards)
- IoU Loss       : directly optimises the eval metric (polygon IoU)
- DT Boundary    : MaxPool-based boundary region (±3px) with 5× weight
- Sobel Boundary : gradient-based edge loss as auxiliary signal
- LossWeightScheduler : warmup → gradually increase IoU/boundary weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Combined segmentation + classification loss for ROI detection.

    Parameters
    ----------
    None — all loss weights are passed at forward time via LossWeightScheduler.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Static loss components
    # ------------------------------------------------------------------

    @staticmethod
    def _vectorized_ohem(loss_map: torch.Tensor, tgt_flat: torch.Tensor,
                         focal_gamma: float = 2.0) -> torch.Tensor:
        """
        Vectorized OHEM + Focal Weighting.

        Removes B×C for-loop from original OHEM implementation.
        Automatically focuses on hard pixels (boundary, small cards).

        loss_map : (B, C, N)  per-pixel MSE
        tgt_flat : (B, C, N)  ground truth
        """
        B, C, N = loss_map.shape
        pos_mask = tgt_flat > 0
        neg_mask = ~pos_mask

        # Focal weighting: harder pixels get higher weight
        focal_weight = loss_map.detach().clamp(0, 1) ** (focal_gamma / 2.0) + 1.0
        weighted_loss = loss_map * focal_weight

        # Positive: all positive pixels
        pos_loss_sum = (weighted_loss * pos_mask.float()).sum(dim=-1)   # (B, C)
        n_pos = pos_mask.float().sum(dim=-1).clamp(min=1)

        # Negative OHEM: top-k hard negatives (k = 3 × n_pos)
        neg_loss = weighted_loss.clone()
        neg_loss[pos_mask] = -1.0

        n_neg = neg_mask.float().sum(dim=-1).clamp(min=1)
        k     = (3 * n_pos).clamp(max=n_neg).long()
        max_k = k.max().item()

        neg_sorted, _ = neg_loss.sort(dim=-1, descending=True)
        neg_topk = neg_sorted[:, :, :max_k]

        idx    = torch.arange(max_k, device=loss_map.device).view(1, 1, -1)
        k_mask = idx < k.unsqueeze(-1)
        neg_topk = neg_topk * k_mask.float()

        neg_loss_sum  = neg_topk.sum(dim=-1)
        n_neg_used    = k_mask.float().sum(dim=-1).clamp(min=1)

        channel_loss = (pos_loss_sum + neg_loss_sum) / (n_pos + n_neg_used)
        return channel_loss.mean()

    @staticmethod
    def _dice_loss(pred_flat: torch.Tensor, tgt_flat: torch.Tensor) -> torch.Tensor:
        """Dice loss — region overlap optimisation."""
        intersection = (pred_flat * tgt_flat).sum(dim=-1)
        loss = 1 - (2 * intersection + 1) / (
            pred_flat.sum(dim=-1) + tgt_flat.sum(dim=-1) + 1)
        return loss.mean()

    @staticmethod
    def _iou_loss(pred_flat: torch.Tensor, tgt_flat: torch.Tensor,
                  smooth: float = 1.0) -> torch.Tensor:
        """
        IoU Loss — directly optimises the evaluation metric (polygon IoU).
        Complements Dice loss by penalising union more aggressively.
        """
        intersection = (pred_flat * tgt_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + tgt_flat.sum(dim=-1) - intersection
        return (1 - (intersection + smooth) / (union + smooth)).mean()

    @staticmethod
    def _dt_boundary_loss(pred: torch.Tensor, target: torch.Tensor,
                          boundary_width: int = 3) -> torch.Tensor:
        """
        Distance-Transform Boundary Loss.

        Extracts boundary region (±boundary_width px) via MaxPool erosion/dilation
        and applies 5× weight to pixels near card edges.
        Provides explicit control over boundary width vs Sobel.

        pred, target : (B, 1, H, W)
        """
        ks = boundary_width * 2 + 1
        dilated  = F.max_pool2d(target, ks, stride=1, padding=boundary_width)
        eroded   = -F.max_pool2d(-target, ks, stride=1, padding=boundary_width)
        boundary = dilated - eroded          # 1 near boundary, 0 elsewhere
        weight   = 1.0 + boundary * 4.0     # 5× at boundary
        return ((pred - target) ** 2 * weight).mean()

    @staticmethod
    def _sobel_boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Sobel gradient MSE — auxiliary edge sharpness loss.
        pred, target : (B, 1, H, W)
        """
        kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                           dtype=pred.dtype, device=pred.device).unsqueeze(0)
        ky = kx.transpose(-2, -1)
        gx_p = F.conv2d(pred,   kx, padding=1)
        gy_p = F.conv2d(pred,   ky, padding=1)
        gx_t = F.conv2d(target, kx, padding=1)
        gy_t = F.conv2d(target, ky, padding=1)
        edge_p = (gx_p ** 2 + gy_p ** 2 + 1e-8).sqrt()
        edge_t = (gx_t ** 2 + gy_t ** 2 + 1e-8).sqrt()
        return F.mse_loss(edge_p, edge_t)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, seg_pred: torch.Tensor, seg_target: torch.Tensor,
                cls_pred: torch.Tensor, cls_target: torch.Tensor,
                loss_weights: dict = None) -> torch.Tensor:
        """
        Parameters
        ----------
        seg_pred   : (B, C, H, W)  sigmoid output
        seg_target : (B, C, H, W)  ground truth maps
        cls_pred   : (B, num_classes)
        cls_target : (B,) long
        loss_weights : dict from LossWeightScheduler.step() or None

        Returns
        -------
        torch.Tensor  scalar total loss
        """
        if loss_weights is None:
            loss_weights = {
                'pixel': 1.0, 'dice': 0.5, 'iou': 0.0,
                'boundary': 0.5, 'class': 0.1, 'focal_gamma': 2.0,
            }

        seg_target = F.interpolate(
            seg_target.float(), size=seg_pred.shape[-2:],
            mode='bilinear', align_corners=False)

        pred_flat = seg_pred.flatten(2)       # (B, C, N)
        tgt_flat  = seg_target.flatten(2)

        pixel_loss    = self._vectorized_ohem(
            (pred_flat - tgt_flat) ** 2, tgt_flat,
            focal_gamma=loss_weights['focal_gamma'])
        dice_loss     = self._dice_loss(pred_flat, tgt_flat)
        iou_loss      = self._iou_loss(pred_flat, tgt_flat)
        boundary_loss = self._dt_boundary_loss(seg_pred[:, 0:1], seg_target[:, 0:1])
        sobel_loss    = self._sobel_boundary_loss(seg_pred[:, 0:1], seg_target[:, 0:1])
        class_loss    = self.ce_loss(cls_pred, cls_target)

        return (loss_weights['pixel']    * pixel_loss
                + loss_weights['dice']   * dice_loss
                + loss_weights['iou']    * iou_loss
                + loss_weights['boundary'] * boundary_loss
                + 0.5                    * sobel_loss
                + loss_weights['class']  * class_loss)


class LossWeightScheduler:
    """
    Dynamically adjusts loss weights during training.

    Schedule
    --------
    Early  : pixel loss dominant → stable convergence
    Later  : IoU / boundary weight increases → precision refinement

    Usage
    -----
        scheduler = LossWeightScheduler(total_steps=100000)
        for step in range(total_steps):
            weights = scheduler.step()
            loss = criterion(pred, target, loss_weights=weights)
    """

    def __init__(self, total_steps: int, warmup_ratio: float = 0.1):
        self.total_steps   = total_steps
        self.warmup_steps  = int(total_steps * warmup_ratio)
        self.current_step  = 0

    def step(self) -> dict:
        t        = min(self.current_step / max(self.total_steps, 1), 1.0)
        warmup_t = min(self.current_step / max(self.warmup_steps, 1), 1.0)
        self.current_step += 1
        return {
            'pixel'      : 1.0,
            'dice'       : 0.5 + 0.5 * warmup_t,    # 0.5 → 1.0
            'iou'        : 0.5 * warmup_t,            # 0.0 → 0.5
            'boundary'   : 0.5 + 0.5 * t,            # 0.5 → 1.0
            'class'      : 0.1,
            'focal_gamma': 2.0,
        }