import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .basic import *

# ---------------------------------------------------------------------------
# SparseObj: obj_classes (B, Q, C+1), obj_masks (B, Q, N)
#             vs annotation (B, N, C) + optional objects
# ---------------------------------------------------------------------------

class ObjectsLoss(nn.Module):
    """Combined classification + segmentation loss with Hungarian matching.

    For each sample in the batch, converts annotation data to GT segments,
    matches Q predictions to M ground-truth segments via Hungarian matching,
    then computes cross-entropy for classes and BCE + dice for masks.
    Unmatched queries are assigned the no-object class. Supports B>=1.

    Accepts the same ``annotation`` format as ``AnnotationLoss`` plus
    optional ``objects`` from the collector. Internally converts to
    ``gt_classes (M,)`` + ``gt_masks (M, N)`` via ``_annotation_to_detr_targets``.

    The loss formulation is native to DETR and hardcoded:
        - **CE** on per-query class predictions (Q classes + no-object).
        - **BCE** on per-query mask logits for matched queries.
        - **Dice** on per-query mask logits for matched queries.

    Args:
        ann_n_classes (int): Number of semantic classes (excluding no-object).
        cost_class (float): Matching cost weight for classification. Default 1.0.
        cost_mask (float): Matching cost weight for mask BCE. Default 1.0.
        cost_dice (float): Matching cost weight for mask dice. Default 1.0.
        no_object_weight (float): CE weight for the no-object class. Default 0.1.
        ce_weight (float): Loss weight for classification CE term. Default 1.0.
        bce_weight (float): Loss weight for mask BCE term. Default 5.0.
        dice_weight (float): Loss weight for mask dice term. Default 2.0.
        threshold (float): Activation threshold for deriving segments from
            annotation when ``objects`` is not provided. Default 0.5.

    Input:
        outputs (dict): SparseObj output with:
            - ``obj_classes`` ``(B, Q, C+1)``
            - ``obj_masks``   ``(B, Q, N)``
        annotation (torch.Tensor): ``(B, N, C)`` per-tile class fractions.
        objects (list[dict], optional): Annotation bags with ``'label'`` and ``'ids'``.

    Output:
        dict: ``loss`` (scalar), ``ce_loss`` (scalar), ``bce_loss`` (scalar),
            ``dice_loss`` (scalar).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        no_object_weight: float = 0.05,
        ce_weight: float = 1.0,
        bce_weight: float = 5.0,
        dice_weight: float = 2.0,
        threshold: float = 0.5,
        ann_cls_weights: dict = None,
    ):
        super().__init__()
        self.matcher = HungarianMatcher(cost_class, cost_mask, cost_dice)
        self.no_object_weight = no_object_weight
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold
        w = weights_tensor(ann_cls_weights)
        self.register_buffer('cls_weight', w)

    def _forward_single(self, pred_cls, pred_msk, annotation_b, objects_b=None):
        """Compute loss for a single sample (no batch dim).

        Args:
            pred_cls (torch.Tensor): (Q, C+1) predicted class logits.
            pred_msk (torch.Tensor): (Q, N) predicted mask logits.
            annotation_b (torch.Tensor): (N, C) per-tile class fractions.
            objects_b: Annotation bags for this sample (or None).

        Returns:
            tuple: (ce_loss, bce_loss, dice_loss) scalars.
        """
        device = pred_cls.device
        ann_n_classes = pred_cls.shape[1] - 1

        # Convert annotation to DETR gt format
        gt_classes, gt_masks = self._annotation_to_detr_targets(
            annotation_b.unsqueeze(0), objects_b)
        gt_classes = gt_classes.to(device)
        gt_masks = gt_masks.to(device)

        # Hungarian matching
        pred_idx, gt_idx = self.matcher(pred_cls, pred_msk, gt_classes, gt_masks)
        pred_idx = pred_idx.to(device)
        gt_idx = gt_idx.to(device)

        # --- Classification loss ---
        no_obj_class = ann_n_classes
        target_cls = torch.full((pred_cls.size(0),), no_obj_class, dtype=torch.long, device=device)
        if len(pred_idx) > 0:
            target_cls[pred_idx] = gt_classes[gt_idx]

        weight = torch.ones(ann_n_classes + 1, device=device)
        if self.cls_weight is not None:
            weight[:ann_n_classes] = self.cls_weight.to(device)
        weight[no_obj_class] = self.no_object_weight
        ce_loss = F.cross_entropy(pred_cls, target_cls, weight=weight)

        # --- Mask losses (only for matched queries) ---
        if len(pred_idx) > 0:
            matched_pred = pred_msk[pred_idx]       # (K, N)
            matched_gt = gt_masks[gt_idx].float()   # (K, N)

            bce_loss = F.binary_cross_entropy_with_logits(matched_pred, matched_gt)

            pred_sig = matched_pred.sigmoid()
            num = 2 * (pred_sig * matched_gt).sum(-1)
            den = pred_sig.sum(-1) + matched_gt.sum(-1) + 1e-6
            dice_loss = (1 - num / den).mean()
        else:
            bce_loss = torch.tensor(0.0, device=device)
            dice_loss = torch.tensor(0.0, device=device)

        return ce_loss, bce_loss, dice_loss

    def forward(self, obj_classes: torch.Tensor, obj_masks: torch.Tensor, annotation: torch.Tensor, objects=None, **kwargs) -> dict:
        """Compute losses averaged over the batch.

        Args:
            outputs (dict): Model output with ``obj_classes`` (B, Q, C+1)
                and ``obj_masks`` (B, Q, N).
            annotation (torch.Tensor): ``(B, N, C)`` per-tile class fractions.
            objects (list, optional): Annotation bags from collector.
            **kwargs: Ignored (allows passing full data dict).

        Returns:
            dict: ``loss``, ``ce_loss``, ``bce_loss``, ``dice_loss``.
        """
        B = obj_classes.size(0)

        total_ce = 0.0
        total_bce = 0.0
        total_dice = 0.0
        for b in range(B):
            objects_b = objects[b] if objects is not None and len(objects) > b else None
            ce, bce, dice = self._forward_single(
                obj_classes[b], obj_masks[b],
                annotation[b], objects_b)
            total_ce = total_ce + ce
            total_bce = total_bce + bce
            total_dice = total_dice + dice

        ce_loss = total_ce / B
        bce_loss = total_bce / B
        dice_loss = total_dice / B
        obj_loss = self.ce_weight * ce_loss + self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return {
            "obj_loss": obj_loss,
            "ce_loss": ce_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
        }
        
    def _annotation_to_detr_targets(self, annotation, objects=None):
        """Convert annotation data to DETR ground-truth format.

        Two modes:

        1. **With objects** — each bag defines one GT segment directly:
        ``gt_classes[m]`` = argmax of bag ``label``,
        ``gt_masks[m, n]`` = 1 if tile n is in bag m's ``ids``.

        2. **Without objects** — derive segments from ``annotation (N, C)``:
        for each class c where any tile exceeds ``threshold``, create one
        segment with ``gt_classes[m] = c`` and ``gt_masks[m, :]`` = binary
        mask of tiles with ``annotation[:, c] > threshold``.

        Args:
            annotation (torch.Tensor): ``(1, N, C)`` or ``(N, C)`` per-tile class fractions.
            objects (list[dict], optional): Annotation bags from collector, each with
                ``'label'`` (list of length C) and ``'ids'`` (list/array of tile indices).
            threshold (float): Activation threshold for deriving segments from
                annotation when ``objects`` is not provided. Default 0.5.

        Returns:
            tuple: ``(gt_classes, gt_masks)``
                - ``gt_classes`` ``(M,)`` long tensor of class indices.
                - ``gt_masks``   ``(M, N)`` float tensor of binary masks.
        """
        # Normalise to (N, C)
        if annotation.dim() == 3:
            annotation = annotation[0]  # (N, C)
        N, C = annotation.shape
        device = annotation.device

        if objects is not None and len(objects) > 0:
            # Each bag is one GT segment
            gt_classes_list = []
            gt_masks_list = []
            for bag in objects:
                label = bag['label']  # list of length C (multi-class)
                ids = bag['ids']      # list/array of tile indices
                if len(ids) == 0:
                    continue
                # Class = argmax of the bag label
                cls_idx = int(torch.tensor(label).argmax())
                gt_classes_list.append(cls_idx)
                mask = torch.zeros(N, device=device)
                mask[ids] = 1.0
                gt_masks_list.append(mask)

            if len(gt_classes_list) == 0:
                return (torch.tensor([], dtype=torch.long, device=device),
                        torch.zeros(0, N, device=device))

            gt_classes = torch.tensor(gt_classes_list, dtype=torch.long, device=device)
            gt_masks = torch.stack(gt_masks_list, dim=0)  # (M, N)
        else:
            # Derive segments from annotation: one segment per active class
            gt_classes_list = []
            gt_masks_list = []
            for c in range(C):
                active = (annotation[:, c] > self.threshold).float()  # (N,)
                if active.sum() > 0:
                    gt_classes_list.append(c)
                    gt_masks_list.append(active)

            if len(gt_classes_list) == 0:
                return (torch.tensor([], dtype=torch.long, device=device),
                        torch.zeros(0, N, device=device))

            gt_classes = torch.tensor(gt_classes_list, dtype=torch.long, device=device)
            gt_masks = torch.stack(gt_masks_list, dim=0)  # (M, N)

        return gt_classes, gt_masks

class HungarianMatcher:
    """Bipartite matcher between predicted queries and ground-truth segments.

    Computes a cost matrix combining classification, mask BCE, and mask dice
    costs, then solves the optimal assignment with ``scipy.linear_sum_assignment``.

    Args:
        cost_class (float): Weight for classification cost.
        cost_mask (float): Weight for binary cross-entropy mask cost.
        cost_dice (float): Weight for dice mask cost.
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def __call__(self, obj_classes, obj_masks, gt_classes, gt_masks):
        """Compute optimal assignment.

        Args:
            obj_classes (torch.Tensor): (Q, C+1) predicted class logits.
            obj_masks (torch.Tensor): (Q, N) predicted mask logits.
            gt_classes (torch.Tensor): (M,) ground-truth class indices.
            gt_masks (torch.Tensor): (M, N) ground-truth binary masks.

        Returns:
            tuple: (pred_indices, gt_indices) — matched index arrays.
        """
        Q = obj_classes.size(0)
        M = gt_classes.size(0)

        if M == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Classification cost: negative log-prob of the target class
        prob = obj_classes.softmax(-1)                     # (Q, C+1)
        cost_cls = -prob[:, gt_classes]                     # (Q, M)

        # Mask BCE cost
        pred_sigmoid = obj_masks.sigmoid()                 # (Q, N)
        cost_bce = F.binary_cross_entropy_with_logits(
            obj_masks.unsqueeze(1).expand(-1, M, -1),      # (Q, M, N)
            gt_masks.unsqueeze(0).expand(Q, -1, -1),        # (Q, M, N)
            reduction='none',
        ).mean(-1)                                          # (Q, M)

        # Mask dice cost
        num = 2 * (pred_sigmoid.unsqueeze(1) * gt_masks.unsqueeze(0)).sum(-1)
        den = pred_sigmoid.unsqueeze(1).sum(-1) + gt_masks.unsqueeze(0).sum(-1) + 1e-6
        cost_dice = 1 - num / den                           # (Q, M)

        # Total cost
        C = (self.cost_class * cost_cls +
             self.cost_mask * cost_bce +
             self.cost_dice * cost_dice)

        row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
        return torch.tensor(row_ind, dtype=torch.long), torch.tensor(col_ind, dtype=torch.long)

