"""Metric classes for downstream model predictions.

Organised to mirror the loss taxonomy in ``losser.py``:

1. **Mutually-exclusive** (``hard_ce``, ``focal``, ``soft_ce``, ``kl_div``):
   ``ExclusiveLabelMetrics`` — argmax accuracy, macro-F1, per-class accuracy.

2. **Regression** (``mse``, ``mae``):
   ``RegressionLabelMetrics`` — per-class MAE and MSE.

3. **Multi-label** (``bce``, ``multi_fc``):
   ``MultiLabelMetrics`` — per-class accuracy (threshold 0.5), per-class F1,
   overall Hamming accuracy.

4. **Annotation / per-tile** (UNet, DETR):
   ``AnnotationMetrics`` — tile accuracy (argmax), per-class Dice, per-class IoU.

All metric classes accept ``(pred, target)`` tensors already on any device and
internally call ``.detach().cpu()`` so they are safe to call inside training
loops with live gradients.

Usage::

    m = ExclusiveLabelMetrics()
    result = m(pred, target)   # dict of scalar floats

``compute_metrics`` dispatches on prediction keys and the registered metric
objects attached to each model head.
"""

import torch
import torch.nn as nn
import numpy as np
import collections
from typing import Literal

from apeiron.utils import to_cpu


# ============================================================================
# Mapping: loss_type -> metric class name (used in choose_inferencer)
# ============================================================================

METRIC_MODES = {
    'hard_ce':  'exclusive',
    'focal':    'exclusive',
    'soft_ce':  'exclusive',
    'kl_div':   'exclusive',
    'mse':      'regression',
    'mae':      'regression',
    'bce':      'multilabel',
    'multi_fc': 'multilabel',
}


# ============================================================================
# Metric classes
# ============================================================================

class LabelMetrics:
    def __init__(self, mode: str = 'exclusive'):
        super().__init__()
        self.mode = mode

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        self.threshold = threshold

        if self.mode == 'multilabel':
            return self.multilabel(pred, target)
        elif self.mode == 'exclusive':
            return self.exclusive(pred, target)
        elif self.mode == 'regression':
            return self.regression(pred, target)
    
    @staticmethod
    def _per_class_f1(pred_bin: torch.Tensor, tgt_bin: torch.Tensor) -> dict:
        """Per-class F1 from binary (B, C) tensors. Returns dict f1_c0, f1_c1, ..."""
        C = pred_bin.size(-1)
        out = {}
        for c in range(C):
            p = pred_bin[:, c]
            t = tgt_bin[:, c]
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            denom = 2 * tp + fp + fn
            out[f'f1_c{c}'] = (2 * tp / denom).item() if denom > 0 else 0.0
        return out

    @torch.no_grad()
    def multilabel(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Metrics for independent per-class (multi-label) predictions.

        Suitable for ``bce``, ``multi_fc`` losses. Each class is thresholded
        independently at 0.5.

        Input:
            pred   (torch.Tensor): ``(B, C)`` sigmoid probabilities.
            target (torch.Tensor): ``(B, C)`` binary labels (each class in {0, 1}).

        Output:
            dict:
                - ``hamming_acc``  — fraction of (sample, class) pairs correct.
                - ``f1_macro``     — macro-averaged F1 across classes.
                - ``f1_c{i}``      — per-class F1.
                - ``acc_c{i}``     — per-class accuracy.
        """
        target = to_cpu(target)
        pred_bin = (pred >= self.threshold).float()     # (B, C)
        tgt_bin = (target >= self.threshold).float()    # (B, C)

        hamming_acc = (pred_bin == tgt_bin).float().mean().item()

        per_f1 = self._per_class_f1(pred_bin, tgt_bin)
        f1_macro = float(np.mean(list(per_f1.values())))

        C = pred.size(-1)
        per_acc = {f'acc_c{c}': (pred_bin[:, c] == tgt_bin[:, c]).float().mean().item()
                   for c in range(C)}

        return {'hamming_acc': hamming_acc, 'f1_macro': f1_macro, **per_f1, **per_acc}

    @torch.no_grad()
    def exclusive(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Metrics for mutually-exclusive label predictions.

        Suitable for ``hard_ce``, ``focal``, ``soft_ce``, ``kl_div`` losses.
        Treats the argmax of both pred and target as the single true class.

        Input:
            pred   (torch.Tensor): ``(B, C)`` class probabilities.
            target (torch.Tensor): ``(B, C)`` one-hot or soft labels.

        Output:
            dict:
                - ``accuracy``     — overall argmax accuracy.
                - ``f1_macro``     — macro-averaged F1 across classes.
                - ``f1_c{i}``      — per-class F1.
                - ``acc_c{i}``     — per-class accuracy (class i vs rest).
        """
        target = to_cpu(target)
        pred_cls = pred.argmax(dim=-1)      # (B,)
        true_cls = target.argmax(dim=-1)    # (B,)
        C = pred.size(-1)

        accuracy = (pred_cls == true_cls).float().mean().item()

        # Per-class: treat as one-vs-rest binary
        pred_bin = torch.zeros_like(pred)
        pred_bin.scatter_(-1, pred_cls.unsqueeze(-1), 1.0)   # (B, C) one-hot
        tgt_bin = torch.zeros_like(target)
        tgt_bin.scatter_(-1, true_cls.unsqueeze(-1), 1.0)    # (B, C) one-hot

        per_f1 = self._per_class_f1(pred_bin, tgt_bin)
        f1_macro = float(np.mean(list(per_f1.values())))

        per_acc = {}
        for c in range(C):
            per_acc[f'acc_c{c}'] = ((pred_cls == c) == (true_cls == c)).float().mean().item()

        return {'accuracy': accuracy, 'f1_macro': f1_macro, **per_f1, **per_acc}

    @torch.no_grad()
    def regression(self, pred: torch.Tensor, label: torch.Tensor) -> dict:
        """Metrics for regression score predictions.

        Suitable for ``mse``, ``mae`` losses.

        Input:
            pred   (torch.Tensor): ``(B, C)`` predicted scores.
            label (torch.Tensor): ``(B, C)`` ground-truth scores.

        Output:
            dict:
                - ``mae``      — overall mean absolute error.
                - ``mse``      — overall mean squared error.
                - ``mae_c{i}`` — per-class MAE.
                - ``mse_c{i}`` — per-class MSE.
        """
        label = to_cpu(label)
        diff = pred - label                # (B, C)
        mae = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()

        per_mae = {f'mae_c{c}': diff[:, c].abs().mean().item() for c in range(pred.size(-1))}
        per_mse = {f'mse_c{c}': (diff[:, c] ** 2).mean().item() for c in range(pred.size(-1))}

        return {'mae': mae, 'mse': mse, **per_mae, **per_mse}


class AnnotationMetrics:
    """Metrics for per-tile annotation predictions (UNet / DETR).

    Args:
        mode (str): How to compute ``tile_acc``.
            - ``'exclusive'`` — argmax (mutually exclusive classes per tile).
            - ``'multilabel'`` — hamming accuracy at threshold (independent classes).
            Default ``'exclusive'``.
        threshold (float): Binarisation threshold for Dice, IoU, and multilabel
            tile_acc. Default ``0.5``.

    Input:
        pred   (torch.Tensor): ``(B, N, C)`` per-tile probabilities.
        target (torch.Tensor): ``(B, N, C)`` per-tile labels.

    Output:
        dict:
            - ``tile_acc``    — tile accuracy (argmax or hamming depending on mode).
            - ``dice_macro``  — macro-averaged Dice across classes.
            - ``iou_macro``   — macro-averaged IoU across classes.
            - ``dice_c{i}``   — per-class Dice.
            - ``iou_c{i}``    — per-class IoU.
    """

    def __init__(self, mode: str = 'exclusive'):
        super().__init__()
        self.mode = mode

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, annotation: torch.Tensor, threshold: float = 0.5) -> dict:
        self.threshold = threshold
        annotation = to_cpu(annotation)

        pred_bin = (pred >= self.threshold).float()     # (B, N, C)
        tgt_bin = (annotation >= self.threshold).float()    # (B, N, C)

        if self.mode == 'exclusive':
            pred_cls = pred.argmax(dim=-1)              # (B, N)
            true_cls = annotation.argmax(dim=-1)            # (B, N)
            tile_acc = (pred_cls == true_cls).float().mean().item()
        else:
            tile_acc = (pred_bin == tgt_bin).float().mean().item()

        per_dice = self._per_class_dice(pred_bin, tgt_bin)
        per_iou = self._per_class_iou(pred_bin, tgt_bin)
        dice_macro = float(np.mean(list(per_dice.values())))
        iou_macro = float(np.mean(list(per_iou.values())))

        return {'tile_acc': tile_acc,
                'dice_macro': dice_macro, 'iou_macro': iou_macro,
                **per_dice, **per_iou}

    @staticmethod
    def _per_class_dice(pred_bin: torch.Tensor, tgt_bin: torch.Tensor) -> dict:
        """Per-class Dice from (B, N, C) binary tensors. Returns dict dice_c0, ..."""
        C = pred_bin.size(-1)
        out = {}
        for c in range(C):
            p = pred_bin[..., c]            # (B, N)
            t = tgt_bin[..., c]             # (B, N)
            inter = (p * t).sum(dim=-1)     # (B,)
            denom = p.sum(dim=-1) + t.sum(dim=-1)
            out[f'dice_c{c}'] = ((2 * inter) / (denom + 1e-6)).mean().item()
        return out

    @staticmethod
    def _per_class_iou(pred_bin: torch.Tensor, tgt_bin: torch.Tensor) -> dict:
        """Per-class IoU from (B, N, C) binary tensors. Returns dict iou_c0, ..."""
        C = pred_bin.size(-1)
        out = {}
        for c in range(C):
            p = pred_bin[..., c]
            t = tgt_bin[..., c]
            inter = (p * t).sum(dim=-1)
            union = ((p + t) >= 1).float().sum(dim=-1)
            out[f'iou_c{c}'] = (inter / (union + 1e-6)).mean().item()
        return out


class ObjectsMetrics:
    """Metrics for DETR-style object detection predictions.

    Evaluates ``pred_obj`` ``(B, Q, N, C)`` against ground-truth objects or
    annotation. Queries are greedily matched to GT segments by mask IoU
    (highest-IoU-first). Unmatched queries are counted as false positives.

    Input:
        pred (dict):
            - ``pred_obj``      ``list[list[dict]]`` — per-query objects with 'ids', 'labels', 'scores'.
        target (dict):
            - ``objects`` (list[list[dict]]): GT objects per sample, each dict
              with ``'label'`` (list[float], length C) and ``'ids'`` (tile indices).
              **or**
            - ``annotation`` ``(B, N, C)`` per-tile class fractions (fallback).
        threshold (float): Confidence threshold for counting a query as a
            detection. Default ``0.5``.

    Output:
        dict:
            - ``det_cls_acc``   — fraction of matched queries with correct class.
            - ``mask_iou``      — mean mask IoU over matched query-GT pairs.
            - ``mask_dice``     — mean mask Dice over matched query-GT pairs.
            - ``recall``        — fraction of GT objects matched (IoU ≥ iou_threshold).
            - ``precision``     — fraction of active queries that are TP.
            - ``ap``            — average precision (area under precision-recall curve).
            - ``mask_iou_c{i}`` — per-class mean mask IoU.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def __call__(self,
                 pred_obj: list,
                 objects=None,
                 annotation=None,
                 threshold: float = 0.5,
                 top_k_factor: int = 2,
                 min_queries: int = 5) -> dict:
        
        annotation = to_cpu(annotation) if annotation is not None else None

        B = len(pred_obj)
        # N and C must be inferred from the data
        # Fallbacks if annotation is None and we need dimensions
        N = annotation.shape[1] if annotation is not None else 0
        C = annotation.shape[2] if annotation is not None else 0

        # Attempt to figure out N and C if we don't have annotation
        if N == 0 or C == 0:
            for b in range(B):
                for obj in pred_obj[b]:
                    C = len(obj['labels'])
                    if len(obj['ids']) > 0:
                        N = max(N, max(obj['ids']) + 1)
                if objects and objects[b]:
                    for obj in objects[b]:
                        C = len(obj['label'])
                        if len(obj['ids']) > 0:
                            N = max(N, max(obj['ids']) + 1)

        # Accumulate over batch
        all_iou, all_dice, all_cls_correct = [], [], []
        gt_matched_total, gt_total = 0, 0
        tp_scores, fp_scores = [], []   # for AP curve
        per_class_iou = {c: [] for c in range(C)}

        for b in range(B):
            pred_b = pred_obj[b]       # list[dict]
            
            # --- Build GT masks & class labels ---
            objects_b = (objects or [None] * B)[b]
            gt_masks, gt_classes = self._build_gt(objects_b, annotation, b, N, C)
            # gt_masks:   (M, N) binary float
            # gt_classes: (M,)   int

            M = len(gt_classes)
            gt_total += M
            
            # Number of queries available
            Q = len(pred_b)

            # Cap to top-K queries — queries are already ranked by construction
            K = max(min_queries, M * top_k_factor)
            K = min(K, Q)  
            pred_b = pred_b[:K]

            if M == 0:
                # All active queries are FPs
                for q in range(K):
                    if pred_b[q]['scores'] >= threshold:
                        fp_scores.append(pred_b[q]['scores'])
                continue

            if K == 0:
                continue

            # Reconstruct binary masks and classes for greedy matching
            pred_bin = torch.zeros(K, N)
            query_cls = torch.zeros(K, dtype=torch.long)
            scores_b = torch.zeros(K)
            
            for q in range(K):
                ids = pred_b[q]['ids']
                if len(ids) > 0:
                    pred_bin[q, ids] = 1.0
                query_cls[q] = int(np.argmax(pred_b[q]['labels']))
                scores_b[q] = pred_b[q]['scores']

            # IoU matrix: (Q, M)
            iou_matrix = self._mask_iou_matrix(pred_bin, gt_masks)  # (K, M)

            matched_gt = set()
            matched_q  = set()
            # Sort by IoU descending for greedy match
            flat_iou = iou_matrix.reshape(-1)
            sorted_idx = flat_iou.argsort(descending=True)

            for idx in sorted_idx:
                q_idx = int(idx // M)
                m_idx = int(idx % M)
                if q_idx in matched_q or m_idx in matched_gt:
                    continue
                if flat_iou[idx] < self.iou_threshold:
                    break
                matched_q.add(q_idx)
                matched_gt.add(m_idx)

                iou_val  = iou_matrix[q_idx, m_idx].item()
                dice_val = self._dice(pred_bin[q_idx], gt_masks[m_idx]).item()
                cls_ok   = int(query_cls[q_idx].item() == gt_classes[m_idx])

                all_iou.append(iou_val)
                all_dice.append(dice_val)
                all_cls_correct.append(cls_ok)
                per_class_iou[gt_classes[m_idx]].append(iou_val)

                tp_scores.append(scores_b[q_idx].item())
                gt_matched_total += 1

            # Unmatched active queries within top-K → FP
            active_queries = set((scores_b >= threshold).nonzero(as_tuple=True)[0].tolist())
            fp_queries = active_queries - matched_q
            fp_scores.extend(scores_b[list(fp_queries)].tolist())

        # --- Aggregate ---
        mean_iou  = float(np.mean(all_iou))  if all_iou  else 0.0
        mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
        cls_acc   = float(np.mean(all_cls_correct)) if all_cls_correct else 0.0
        recall    = gt_matched_total / max(gt_total, 1)
        n_active  = len(tp_scores) + len(fp_scores)
        precision = len(tp_scores) / max(n_active, 1)
        ap        = self._compute_ap(tp_scores, fp_scores, gt_total)

        per_class_iou_out = {
            f'mask_iou_c{c}': float(np.mean(v)) if v else 0.0
            for c, v in per_class_iou.items()
        }

        return {
            'det_cls_acc': cls_acc,
            'mask_iou':    mean_iou,
            'mask_dice':   mean_dice,
            'recall':      recall,
            'precision':   precision,
            'ap':          ap,
            **per_class_iou_out,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_gt(self, objects_b, annotation, b, N, C):
        """Return (gt_masks, gt_classes) for one sample.

        Tries ``objects_b`` first, falls back to ``annotation[b]``.
        """
        gt_masks, gt_classes = [], []

        if objects_b is not None and len(objects_b) > 0:
            for obj in objects_b:
                ids   = np.asarray(obj['ids'])
                label = np.asarray(obj['label'])
                if len(ids) == 0:
                    continue
                mask = torch.zeros(N)
                mask[ids] = 1.0
                gt_masks.append(mask)
                gt_classes.append(int(np.argmax(label)))

        elif annotation is not None:
            ann_b = torch.as_tensor(annotation[b]).float()  # (N, C)
            for c in range(C):
                active = (ann_b[:, c] >= 0.5)
                if active.any():
                    mask = active.float()
                    gt_masks.append(mask)
                    gt_classes.append(c)

        if gt_masks:
            return torch.stack(gt_masks, dim=0), gt_classes
        return torch.zeros(0, N), []

    @staticmethod
    def _mask_iou_matrix(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute IoU between all pairs. pred (Q, N), gt (M, N) → (Q, M)."""
        inter = torch.einsum('qn,mn->qm', pred, gt)
        pred_sum = pred.sum(dim=-1, keepdim=True)   # (Q, 1)
        gt_sum   = gt.sum(dim=-1, keepdim=True).t() # (1, M)
        union    = pred_sum + gt_sum - inter
        return inter / (union + 1e-6)

    @staticmethod
    def _dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Dice for a single pair. pred (N,), gt (N,)."""
        inter = (pred * gt).sum()
        return 2 * inter / (pred.sum() + gt.sum() + 1e-6)

    @staticmethod
    def _compute_ap(tp_scores, fp_scores, n_gt: int) -> float:
        """Area under the precision-recall curve."""
        if n_gt == 0:
            return 0.0
        all_scores  = tp_scores + fp_scores
        all_tp_flag = [1] * len(tp_scores) + [0] * len(fp_scores)
        order = np.argsort(all_scores)[::-1]
        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []
        for i in order:
            if all_tp_flag[i]:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)
        # Integrate using trapezoidal rule
        if not recalls:
            return 0.0
        return float(np.trapz(precisions, recalls)) if len(recalls) > 1 else precisions[0]


class TextMetrics:
    """Metrics for text generation (e.g. from Vision-Language Models).

    Evaluates generated strings against target strings.
    Currently implements exact match and basic character-level accuracy
    as placeholders for more complex NLP metrics (ROUGE, BLEU).

    Input:
        pred_texts (list[str]): List of generated strings.
        target_texts (list[str]): List of ground-truth strings.

    Output:
        dict:
            - ``exact_match`` — fraction of perfectly matching strings.
            - ``char_acc``    — average character-level overlap ratio.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def __call__(self, pred_texts: list, target_texts: list) -> dict:
        if pred_texts is None or target_texts is None or len(pred_texts) != len(target_texts):
            return {'exact_match': 0.0, 'char_acc': 0.0}

        exact_matches = 0
        char_accs = []

        for p, t in zip(pred_texts, target_texts):
            p = str(p).strip()
            t = str(t).strip()
            
            if p == t:
                exact_matches += 1
                char_accs.append(1.0)
            else:
                # Basic character overlap (SequenceMatcher could be used for better accuracy)
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, p, t).ratio()
                char_accs.append(ratio)

        return {
            'exact_match': exact_matches / len(target_texts),
            'char_acc': float(np.mean(char_accs)) if char_accs else 0.0
        }


class Metricator:
    """Metric module for slide-level label predictions.

    Wraps the appropriate primitive metric class based on ``loss_type``.
    Also computes attention pseudo-annotation metrics when ``annotation``
    is present in target and ``pred_ann`` is in the prediction dict.

    Args:
        loss_type (str): One of the keys in ``METRIC_MODES``
            (``'hard_ce'``, ``'focal'``, ``'soft_ce'``, ``'kl_div'``,
            ``'mse'``, ``'mae'``, ``'bce'``, ``'multi_fc'``).

    Input:
        predicted (dict): Prediction dict with ``pred_lbl`` ``(B, C)`` and
            optionally ``pred_ann`` ``(B, N, C)``.
        target  (dict): Data batch with ``label`` and optionally ``annotation``.

    Output:
        dict: Flat metric dict of scalar floats.
    """

    def __init__(self, lbl_loss_type: str = None, ann_loss_type: str = None,
                 det_iou_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        if lbl_loss_type:
            self.lbl_metric = LabelMetrics(mode=self.get_metric_mode(lbl_loss_type))

        if ann_loss_type:
            self.ann_metric = AnnotationMetrics(mode=self.get_metric_mode(ann_loss_type))

        self.obj_metric = ObjectsMetrics(iou_threshold=det_iou_threshold)
        self.txt_metric = TextMetrics()

    @staticmethod
    def get_metric_mode(loss_type):
        metric_mode = METRIC_MODES.get(loss_type)
        if metric_mode is None:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from {list(METRIC_MODES)}")
        return metric_mode

    @torch.no_grad()
    def metrics(self, predicted: dict, target: dict, threshold: float = 0.5) -> dict:
        metrics = {}
        label = target.get('label')
        annotation = target.get('annotation')
        objects = target.get('objects')
        text = target.get('text')

        if 'pred_lbl' in predicted and label is not None:
            lbl_t = torch.as_tensor(label).float()
            metrics.update(self.lbl_metric(predicted['pred_lbl'], lbl_t, threshold))

        if 'pred_ann' in predicted and annotation is not None:
            ann_t = torch.as_tensor(annotation).float()
            metrics.update(self.ann_metric(predicted['pred_ann'], ann_t, threshold))

        if all([k in predicted for k in ['pred_obj']]) \
            and any([k is not None for k in [objects, annotation]]):
            ann_t = torch.as_tensor(annotation).float() if annotation is not None else None
            metrics.update(self.obj_metric(
                pred_obj=predicted['pred_obj'],
                objects=objects,
                annotation=ann_t,
                threshold=threshold,
            ))
            
        if 'pred_txt' in predicted and text is not None:
            metrics.update(self.txt_metric(predicted['pred_txt'], text))

        return metrics