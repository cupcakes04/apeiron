"""DETR-style segmentation model for sparse tile features with Hungarian matching.

Consumes ``slide_features_collector`` output:
    - features: (B, N, F) — N tile features of dimension F
    - coords:   (B, N, 2) — spatial (x, y) positions of each tile
    - label:    (B, C)    — C-class soft/hard label

Uses a Transformer encoder-decoder architecture where:
    - Encoder: self-attention over tile features with positional encoding from coords.
    - Decoder: learnable object queries attend to encoded features.
    - Segmentation head: each query predicts a class label + a binary mask over N tiles.

Hungarian matching assigns predicted segments to ground-truth classes.

Produces:
    - obj_classes: (B, Q, C+1) — per-query class logits (including no-object).
    - obj_masks:   (B, Q, N)   — per-query binary mask logits over N tiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from apeiron.utils import to_cpu
from scipy.optimize import linear_sum_assignment
from .helper import apply_pred, ObjectsMetrics, weights_tensor, check_mode

# ---------------------------------------------------------------------------
# Positional encoding from coordinates
# ---------------------------------------------------------------------------

class CoordPositionalEncoding(nn.Module):
    """Learnable positional encoding from 2-D spatial coordinates.

    Projects normalised (x, y) coordinates through an MLP to produce
    positional embeddings of dimension ``embed_dim``.

    Args:
        embed_dim (int): Output positional embedding dimension.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.

        Args:
            coords (torch.Tensor): (B, N, 2) spatial coordinates.

        Returns:
            torch.Tensor: (B, N, D) positional embeddings.
        """
        # Normalise to [0, 1]
        c = coords.float()
        c_min = c.min(dim=1, keepdim=True).values
        c_max = c.max(dim=1, keepdim=True).values
        c_range = (c_max - c_min).clamp(min=1.0)
        c_norm = (c - c_min) / c_range
        return self.mlp(c_norm)

class FourierPositionalEncoding(nn.Module):
    """Fourier feature positional encoding from raw micron coordinates.

    Args:
        embed_dim (int): Output embedding dimension.
        max_freq (int): Maximum frequency for Fourier features.
    """

    def __init__(self, embed_dim: int, max_freq: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.freqs = nn.Parameter(
            torch.linspace(1, max_freq, embed_dim // 4), requires_grad=False
        )
        # MLP after Fourier expansion
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords (torch.Tensor): (B, N, 2) absolute spatial coordinates in microns.
        Returns:
            torch.Tensor: (B, N, D) positional embeddings.
        """
        # coords: (B, N, 2)
        # expand with sin/cos at multiple frequencies
        pos_enc = []
        for f in self.freqs:
            pos_enc.append(torch.sin(coords * f))
            pos_enc.append(torch.cos(coords * f))
        # concatenate along last dimension
        fourier_features = torch.cat(pos_enc, dim=-1)
        return self.mlp(fourier_features)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class SparseDETR(nn.Module):
    """DETR-style model for tile-level segmentation with Hungarian matching.

    Encodes N tile features with positional information from coordinates,
    then decodes Q object queries. Each query predicts a class distribution
    and a binary mask over the N input tiles.

    Supports B>=1 (N must be the same across the batch).

    Args:
        in_features (int): Input feature dimension F.
        ann_n_classes (int): Number of semantic classes C (excluding no-object).
        embed_dim (int): Transformer hidden dimension. Default 256.
        n_heads (int): Number of attention heads. Default 8.
        n_encoder_layers (int): Transformer encoder layers. Default 3.
        n_decoder_layers (int): Transformer decoder layers. Default 3.
        n_queries (int): Number of object queries Q. Default 20.
        dropout (float): Dropout rate. Default 0.1.

    Input:
        features (torch.Tensor): (B, N, F) tile features.
        coords   (torch.Tensor): (B, N, 2) tile spatial coordinates.

    Output:
        dict with:
            - ``obj_classes`` (torch.Tensor): (B, Q, C+1) per-query class logits.
            - ``obj_masks``   (torch.Tensor): (B, Q, N) per-query mask logits.
    """

    def __init__(
        self,
        in_features: int,
        ann_n_classes: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        n_queries: int = 20,
        dropout: float = 0.1,
        
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        no_object_weight: float = 0.05,
        ce_weight: float = 1.0,
        bce_weight: float = 5.0,
        dice_weight: float = 2.0,
        threshold: float = 0.5,
        ann_loss_type: str = 'hard_ce', 
        ann_cls_weights: dict = None,
        det_iou_threshold: float = 0.5, 
    ):
        super().__init__()
        self.ann_n_classes = ann_n_classes
        self.n_queries = n_queries

        # Loss params
        self.matcher = HungarianMatcher(cost_class, cost_mask, cost_dice)
        self.no_object_weight = no_object_weight
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold
        w = weights_tensor(ann_cls_weights)
        self.register_buffer('cls_weight', w)
        self.obj_metric = ObjectsMetrics(iou_threshold=det_iou_threshold)
        self.ann_mode = check_mode(ann_loss_type)

        # Input projection
        self.input_proj = nn.Linear(in_features, embed_dim)

        # Positional encoding from coordinates
        # self.pos_enc = CoordPositionalEncoding(embed_dim)
        self.pos_enc = FourierPositionalEncoding(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Learnable object queries
        self.query_embed = nn.Embedding(n_queries, embed_dim)

        # Prediction heads
        self.class_head = nn.Linear(embed_dim, ann_n_classes + 1)  # +1 for no-object
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor, coords: torch.Tensor, **kwargs) -> dict:
        """Forward pass.

        Args:
            features (torch.Tensor): (B, N, F) tile features.
            coords (torch.Tensor): (B, N, 2) tile coordinates.

        Returns:
            dict: ``obj_classes`` (B, Q, C+1) and ``obj_masks`` (B, Q, N).
        """
        B = features.size(0)

        # 1. Project + positional encoding
        src = self.input_proj(features)         # (B, N, D)
        pos = self.pos_enc(coords)              # (B, N, D)
        src = src + pos

        # 2. Encode
        memory = self.encoder(src)              # (B, N, D)

        # 3. Decode with object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        hs = self.decoder(queries, memory)      # (B, Q, D)

        # 4. Class prediction
        obj_classes = self.class_head(hs)      # (B, Q, C+1)

        # 5. Mask prediction via dot product with memory
        mask_embed = self.mask_head(hs)         # (B, Q, D)
        obj_masks = torch.bmm(mask_embed, memory.permute(0, 2, 1))  # (B, Q, N)

        self.output = {"obj_classes": obj_classes, "obj_masks": obj_masks}
        return self.output


    def _loss_single(self, pred_cls, pred_msk, annotation_b, objects_b=None):
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

    def loss(self, annotation: torch.Tensor,  obj_classes: torch.Tensor=None, obj_masks: torch.Tensor=None, objects=None, **kwargs) -> dict:
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
        if obj_classes is None: obj_classes = self.output.get('obj_classes')
        if obj_masks is None: obj_masks = self.output.get('obj_masks')
        B = obj_classes.size(0)

        total_ce = 0.0
        total_bce = 0.0
        total_dice = 0.0
        for b in range(B):
            objects_b = objects[b] if objects is not None and len(objects) > b else None
            ce, bce, dice = self._loss_single(
                obj_classes[b], obj_masks[b],
                annotation[b], objects_b)
            total_ce = total_ce + ce
            total_bce = total_bce + bce
            total_dice = total_dice + dice

        ce_loss = total_ce / B
        bce_loss = total_bce / B
        dice_loss = total_dice / B
        obj_loss = self.ce_weight * ce_loss + self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return {'objects': {
            "obj_loss": obj_loss,
            "ce_loss": ce_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
        }}
        
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
        
    @torch.no_grad()
    def predict(self, obj_classes: torch.tensor = None, obj_masks: torch.tensor = None, **kwargs) -> dict: 
        '''
        obj_classes: (B, Q, C+1) 
        obj_masks: (B, Q, N)

        returns a list of dict
        
        list({
            "ids": ids.cpu().numpy(),                 # (K,)
            "labels": cls_scores.cpu().tolist(),      # length C
            "scores": float(query_conf.cpu())         # scalar
        })
        '''
        
        if obj_classes is None: obj_classes = self.output.get('obj_classes')
        if obj_masks is None: obj_masks = self.output.get('obj_masks')

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
            
        self.result = {'pred_obj': pred_obj}
        return self.result

    def metric(self, pred_obj: list = None, objects: dict = None, annotation: torch.Tensor = None, threshold: float = 0.5, **kwargs):
        if pred_obj is None: pred_obj = self.result.get('pred_obj')
        ann_t = torch.as_tensor(annotation).float() if annotation is not None else None
        return {'objects': self.obj_metric(
            pred_obj=pred_obj,
            objects=objects,
            annotation=ann_t,
            threshold=threshold,
        )}

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

