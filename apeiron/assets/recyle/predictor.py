"""Prediction transforms for downstream model outputs.

Converts raw model logits into interpretable predictions:

1. **Slide-level** — ``(B, C)`` class probabilities from logits:
    - ``LabelPredictor`` — softmax or sigmoid. Also outputs ``pred_atn``
      ``(N, C)`` when the model provides ``attention`` (e.g. ABMIL).

2. **Annotation-level** — ``(N, C)`` per-tile class probabilities:
    - ``AnnotationPredictor``          — from UNet ``ann_logits (1, C, N)``.
    - ``ObjectsPredictor``      — from DETR ``obj_classes (1, Q, C+1)``
                                         + ``obj_masks (1, Q, N)``.

All predictors accept the raw output dict from the corresponding model
and return a dict with the prediction tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from apeiron.model.downstream import unsqueeze_outer_batch
import numpy as np
from .metricator import Metricator
from apeiron.utils import to_cpu
from typing import Dict, Callable

PRED_TYPES = {
    'hard_ce': 'softmax',
    'focal': 'softmax',
    'soft_ce': 'softmax',
    'kl_div': 'softmax',
    'mse': None,
    'mae': None,
    'bce': 'sigmoid',
    'multi_fc': 'sigmoid',
}

def check_mode(mode):
    if mode not in ('softmax', 'sigmoid'):
        raise ValueError(f"mode must be 'softmax' or 'sigmoid', got '{mode}'")
    return mode

def apply_pred(mode, logits):
    if mode == 'softmax':
        return F.softmax(logits, dim=-1)   # (B, C)
    else:
        return logits.sigmoid()            # (B, C)


# ============================================================================
# (1) Slide-level: logits (B, C) -> (B, C) probabilities
# ============================================================================


class LabelPredictor(Metricator):
    """Convert slide-level logits to class probabilities.

    Works with any model producing ``logits (B, C)`` — ABMIL, TileClassifier,
    or any future model.

    If the model also outputs ``attention (1, N)`` (e.g. ABMIL), the
    attention is transposed and L1-normalised across classes to produce
    ``pred_atn (N, C)`` — a per-tile pseudo-annotation.

    Args:
        mode (str): Activation to apply.
            - ``'softmax'`` — mutually exclusive classes (hard / soft labels).
            - ``'sigmoid'`` — independent per-class (multi-label).
            Default ``'softmax'``.

    Input:
        outputs (dict): Model output with ``logits`` key ``(B, C)``.
            Optionally ``attention`` key ``(1, N)``.

    Output:
        dict:
            - ``pred_lbl`` ``(B, C)`` class probabilities.
            - ``pred_atn``  ``(B, N)`` attention pseudo-annotation (if attention present).
    """

    def __init__(self, lbl_loss_type: str = None, **kwargs):
        super().__init__(lbl_loss_type=lbl_loss_type, **kwargs)
        self.lbl_mode = check_mode(PRED_TYPES[lbl_loss_type])

    @torch.no_grad()
    def __call__(self, lbl_logits: torch.tensor, attention: torch.tensor = None) -> dict:
        # (B, C) # (B, C, N)
        pred_lbl = apply_pred(self.lbl_mode, lbl_logits)
        result = {'pred_lbl': to_cpu(pred_lbl)}

        # If model provides attention maps, convert to per-tile pseudo-annotation
        if attention is not None:         
            result['pred_atn'] = to_cpu(attention)
        return result


# ============================================================================
# (2) Annotation-level: model outputs -> (N, C) per-tile predictions
# ============================================================================


class AnnotationPredictor(Metricator):
    """Convert UNet segmentation logits to per-tile class probabilities.

    Applies sigmoid to ``logits`` and transposes to ``(B, N, C)``.

    Args:
        ann_mode (str): Activation to apply.
            - ``'softmax'`` — mutually exclusive classes per tile.
            - ``'sigmoid'`` — independent per-class per tile (multi-label).
            Default ``'sigmoid'``.

    Input:
        outputs (dict): SparseUNet output with ``logits`` key ``(B, C, N)``.

    Output:
        dict: ``pred_ann`` ``(B, N, C)`` per-tile class probabilities.
    """

    def __init__(self, ann_loss_type: str = None, **kwargs):
        super().__init__(ann_loss_type=ann_loss_type, **kwargs)
        self.ann_mode = check_mode(PRED_TYPES[ann_loss_type])

    @torch.no_grad()
    def __call__(self, ann_logits: torch.tensor) -> dict:
        # (B, N, C)
        pred_ann = apply_pred(self.ann_mode, ann_logits)
        return {'pred_ann': to_cpu(pred_ann)}


# ============================================================================
# (3) Detection-level: model outputs -> (Q, N, C) per-tile predictions
# ============================================================================
class ObjectsPredictor(Metricator):
    """
    Modified to return (B, Q, N, C) per-query per-tile predictions.
    """
    def __init__(self, ann_loss_type: str = None, **kwargs):
        super().__init__(ann_loss_type=ann_loss_type, **kwargs)
        self.ann_mode = check_mode(PRED_TYPES[ann_loss_type])

    @torch.no_grad()
    def __call__(self, obj_classes: torch.tensor, obj_masks: torch.tensor) -> dict: 
        # (B, Q, C+1) # (B, Q, N)

        # 1. Get Class Probs excluding 'no-object' (B, Q, C)
        # We handle the whole batch at once for speed
        class_prob = apply_pred(self.ann_mode, obj_classes)[:, :, :-1]  # (B, Q, C)
        
        # 2. Independent mask probabilities
        mask_prob = torch.sigmoid(obj_masks)  # (B, Q, N)
        
        B, Q, N = mask_prob.shape
        pred_obj = []

        for b in range(B):
            batch_objects = []

            for q in range(Q):
                cls_scores = class_prob[b, q]        # (C,)
                mask_scores = mask_prob[b, q]        # (N,)

                # Query confidence
                query_conf = cls_scores.max()

                # Threshold mask
                ids = torch.nonzero(mask_scores > 0.5).squeeze(1)

                if ids.numel() == 0:
                    continue

                batch_objects.append({
                    "ids": ids.cpu().numpy(),                 # (K,)
                    "labels": cls_scores.cpu().tolist(),      # length C
                    "scores": float(query_conf.cpu())         # scalar
                })

            pred_obj.append(batch_objects)
        return {'pred_obj': pred_obj}



# ============================================================================
# (4) Text-level: model outputs -> ['text1', 'text2', ...]
# ============================================================================
class TextPredictor(Metricator):
    """
    Modified to return (B, Q, N, C) per-query per-tile predictions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def __call__(self, gen_fn: Callable, **kwargs) -> dict:
        return {'pred_txt': gen_fn(**kwargs)}