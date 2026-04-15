import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

# ---------------------------------------------------------------------------
# Generic annotation loss: seg_logits (1, C, N) vs annotation (1, N, C)
# ---------------------------------------------------------------------------

class AnnotationLoss(nn.Module):
    """Generic annotation-level loss for tile segmentation models.

    Applies a per-tile label loss + dice loss to ``seg_logits (1, C, N)``
    against ``annotation (1, N, C)``. B=1 always.

    The annotation labels are transposed to ``(C, N)`` to align with logits,
    then each of the N tiles is treated as an independent sample for the
    chosen label loss, and dice is computed per-class across all N tiles.

    Args:
        loss_type (str): Label loss type for per-tile classification.
            One of ``'hard_ce'``, ``'focal'``, ``'soft_ce'``, ``'kl_div'``,
            ``'mse'``, ``'mae'``, ``'bce'``, ``'multi_fc'``.
        ann_weight (float): Weight for the annotation label loss term. Default 1.0.
        dice_weight (float): Weight for the dice loss term. Default 1.0.
        **kwargs: Forwarded to the underlying label loss constructor.

    Input:
        outputs (dict): Model output with ``logits`` key ``(B, C, N)``.
        annotation (torch.Tensor): ``(B, N, C)`` per-tile class fractions.

    Output:
        dict: ``loss`` (scalar), ``ann_loss`` (scalar), ``dice_loss`` (scalar).
    """

    def __init__(self, ann_loss_type: str = 'bce', ann_weight: float = 1.0,
                 dice_weight: float = 1.0, ann_cls_weights: dict = None, **kwargs):
        super().__init__()
        self.loss_fn = build_label_loss(ann_loss_type, cls_weights=ann_cls_weights, **kwargs)
        self.dice_fn = DiceLoss(cls_weights=ann_cls_weights)
        self.ann_weight = ann_weight
        self.dice_weight = dice_weight

    def forward(self, seg_logits: torch.Tensor, annotation: torch.Tensor, **kwargs) -> dict:
        # logits: (B, N, C), annotation: (B, N, C)
        B = seg_logits.size(0)  # (B, N, C)

        total_tile = 0.0
        total_dice = 0.0
        for b in range(B):
            pred_nc = seg_logits[b]         # (N, C)
            ann_nc = annotation[b]          # (N, C)

            # Per-tile label loss
            total_tile = total_tile + self.loss_fn(pred_nc, ann_nc)
            # Dice loss: transpose to (N, C) for per-class across N tiles
            total_dice = total_dice + self.dice_fn(pred_nc.t(), ann_nc.t())

        tile_loss = total_tile / B
        dice_loss = total_dice / B
        ann_loss = self.ann_weight * tile_loss + self.dice_weight * dice_loss
        return {'ann_loss': ann_loss, 'tile_loss': tile_loss, 'dice_loss': dice_loss}

