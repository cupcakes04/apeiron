import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

# ---------------------------------------------------------------------------
# Generic slide-label loss: logits (B, C) vs label (B, C)
# ---------------------------------------------------------------------------

class LabelLoss(nn.Module):
    """Generic slide-label loss for any model producing ``logits``.

    Works with ABMIL (B=1), TileClassifier (B>=1), or any future model
    whose output dict contains a ``logits`` key of shape ``(B, C)``.

    Args:
        loss_type (str): One of ``'hard_ce'``, ``'focal'``, ``'soft_ce'``,
            ``'kl_div'``, ``'mse'``, ``'mae'``, ``'bce'``, ``'multi_fc'``.
        **kwargs: Forwarded to the underlying loss constructor.

    Input:
        outputs (dict): Model output with ``logits`` key ``(B, C)``.
        label (torch.Tensor): ``(B, C)`` ground-truth label.

    Output:
        dict: ``loss`` (scalar), ``lbl_loss`` (scalar).
    """

    def __init__(self, lbl_loss_type: str = 'hard_ce', lbl_cls_weights: dict = None, **kwargs):
        super().__init__()
        self.lbl_loss_fn = build_label_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)

    def forward(self, lbl_logits: torch.Tensor, label: torch.Tensor, **kwargs) -> dict:
        lbl_loss = self.lbl_loss_fn(lbl_logits, label)
        return {'lbl_loss': lbl_loss}
