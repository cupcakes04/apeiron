"""True Sparse UNet operating on tile features using spconv.

Consumes ``slide_features_collector`` output:
    - features: (B, N, F) — N tile features of dimension F
    - coords:   (B, N, 2) — spatial (x, y) positions of each tile

Produces:
    - ann_logits:  (B, N, C_ann) — per-class segmentation logits at original N positions
    - lbl_logits:  (B, C_lbl)    — slide-level classification logits (attention-pooled)
    - attention:   (B, N)        — normalised attention weights over tiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
# Please install spconv to use SparseUNet: pip install spconv-cu118 (or cu121)

from .helper import GatedAttention, build_loss, check_mode, get_metric_mode, AnnotationMetrics, LabelMetrics, apply_pred
from apeiron.utils import to_cpu

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SparseConvBlock(nn.Module):
    """Double sparse convolution block with unique indice keys for sharing sparsity."""

    def __init__(self, in_ch: int, out_ch: int, indice_key: str):
        super().__init__()
        self.block = spconv.SparseSequential(
            spconv.SubMConv2d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            spconv.SubMConv2d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=f"{indice_key}_2"),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SparseDownBlock(nn.Module):
    """Encoder block: Strided SparseConv -> SparseConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, level: int):
        super().__init__()
        self.down = spconv.SparseSequential(
            spconv.SparseConv2d(in_ch, in_ch, kernel_size=2, stride=2, indice_key=f"down_{level}"),
        )
        self.conv = SparseConvBlock(in_ch, out_ch, indice_key=f"subm_{level+1}")

    def forward(self, x):
        return self.conv(self.down(x))


class SparseUpBlock(nn.Module):
    """Decoder block: InverseSparseConv -> Concat skip -> SparseConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, level: int):
        super().__init__()
        self.up = spconv.SparseInverseConv2d(in_ch, out_ch, kernel_size=2, indice_key=f"down_{level}")
        self.conv = SparseConvBlock(out_ch * 2, out_ch, indice_key=f"subm_{level}_up")

    def forward(self, x, skip):
        x_up = self.up(x)
        # SparseInverseConv2d guarantees the output spatial coords match the original skip exactly
        new_features = torch.cat([skip.features, x_up.features], dim=1)
        x_cat = skip.replace_feature(new_features)
        return self.conv(x_cat)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SparseUNet(nn.Module):
    """True Sparse UNet using spconv for tile-level segmentation with slide classification.

    Preserves exact spatial sparsity. Avoids smudging from dense grids by operating 
    strictly on valid tile locations.

    Args:
        in_features (int): Input feature dimension F.
        ann_n_classes (int): Number of segmentation classes C_ann.
        lbl_n_classes (int): Number of slide-level label classes C_lbl.
        base_channels (int): Channel width of the first encoder stage. Default 256.
        depth (int): Number of encoder downsampling stages. Default 3.
    """

    def __init__(
        self,
        in_features: int,
        ann_n_classes: int = 0,
        lbl_n_classes: int = 0,
        base_channels: int = 512,
        depth: int = 3,
        ann_weight: float = 1.0,
        dice_weight: float = 1.0, 
        ann_loss_type: str = 'bce', 
        ann_cls_weights: dict = None, 
        lbl_loss_type: str = 'hard_ce', 
        lbl_cls_weights: dict = None,
        **kwargs
    ):
        super().__init__()
        self.ann_n_classes = ann_n_classes
        self.lbl_n_classes = lbl_n_classes
        self.depth = depth
        self.bottleneck_ch = base_channels * (2 ** depth)
        
        self.ann_loss_fn = build_loss(ann_loss_type, cls_weights=ann_cls_weights, **kwargs)
        self.dice_fn = build_loss('dice', cls_weights=ann_cls_weights)
        self.ann_weight = ann_weight
        self.dice_weight = dice_weight
        self.ann_metric = AnnotationMetrics(mode=get_metric_mode(ann_loss_type))
        self.ann_mode = check_mode(ann_loss_type)

        # Project input features to base_channels with an MLP
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, base_channels),
            nn.LayerNorm(base_channels),
        )

        # Encoder
        self.enc_first = SparseConvBlock(base_channels, base_channels, indice_key="subm_0")
        self.encoders = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.encoders.append(SparseDownBlock(ch, ch * 2, level=i))
            ch *= 2

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth):
            lvl = depth - 1 - i
            self.decoders.append(SparseUpBlock(ch, ch // 2, level=lvl))
            ch //= 2

        # Segmentation head
        if self.ann_n_classes > 0:
            self.seg_head = nn.Linear(base_channels, ann_n_classes)

        # Classification head (ABMIL) on high-resolution features
        if self.lbl_n_classes > 0:
            self.attention = GatedAttention(base_channels, 256)
            self.classifier = nn.Linear(base_channels, self.lbl_n_classes)
            self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)
            self.lbl_metric = LabelMetrics(mode=get_metric_mode(lbl_loss_type))
            self.lbl_mode = check_mode(lbl_loss_type)

        self.output = {}
        self.result = {}

    def forward(self, features: torch.Tensor, coords: torch.Tensor, coords_size: int, **kwargs) -> dict:
        B, N_tiles, C_in = features.shape

        # 1. Project features
        feats = self.input_proj(features)

        # 2. Compute spatial layout
        unique_indices, inverse_indices, alloc_h, alloc_w = self._compute_spatial_layout(
            coords, coords_size, B, N_tiles, features.device
        )

        feats_flat = feats.view(-1, feats.size(-1))
        
        if unique_indices.size(0) < B * N_tiles:
            count = torch.bincount(inverse_indices, minlength=unique_indices.size(0)).unsqueeze(1).float()
            feats_unique = torch.zeros((unique_indices.size(0), feats.size(-1)), device=features.device)
            feats_unique.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, feats.size(-1)), feats_flat)
            feats_flat_unique = feats_unique / count
        else:
            feats_flat_unique = feats_flat

        # 3. Create True Sparse Tensor
        x_sparse = spconv.SparseConvTensor(
            features=feats_flat_unique,
            indices=unique_indices,
            spatial_shape=[alloc_h, alloc_w],
            batch_size=B
        )

        # 4. UNet encoder
        skips = []
        x_enc = self.enc_first(x_sparse)
        skips.append(x_enc)
        for enc in self.encoders:
            x_enc = enc(x_enc)
            skips.append(x_enc)

        # 5. UNet decoder back to original resolution
        skips.pop()  # bottleneck skip not used
        for dec in self.decoders:
            x_enc = dec(x_enc, skips.pop())

        final_feats = x_enc.features
        result = {}

        # 6. Segment (optional)
        if self.ann_n_classes > 0:
            # Map back to original padded / overlapping grid
            seg_logits_unique = self.seg_head(final_feats) # (N_unique, C_ann)
            seg_logits_flat = seg_logits_unique[inverse_indices] # (B*N, C_ann)
            result['ann_logits'] = seg_logits_flat.view(B, N_tiles, self.ann_n_classes)

        # 7. Classification via High-Res ABMIL
        if self.lbl_n_classes > 0:
            feats_flat_mapped = final_feats[inverse_indices]
            feats_dense = feats_flat_mapped.view(B, N_tiles, -1)
            
            # MIL Pass
            A_weights = self.attention(feats_dense)
            
            # Pool features
            M = (A_weights * feats_dense).sum(dim=1)
            
            lbl_logits = self.classifier(M)
            result['lbl_logits'] = lbl_logits
            result['attention'] = A_weights.squeeze(-1)
            
        self.output = result
        return self.output
        

    def _compute_spatial_layout(self, coords: torch.Tensor, coords_size: int, B: int, N_tiles: int, device: torch.device):
        # 2. Compute spatial layout
        x, y = coords[:, :, 0].float(), coords[:, :, 1].float()
        
        x_mins = x.min(dim=1, keepdim=True).values
        y_mins = y.min(dim=1, keepdim=True).values

        col = ((x - x_mins) / coords_size).round().long()
        row = ((y - y_mins) / coords_size).round().long()

        # Spatial Jitter / Augmentation to prevent fixed-grid memorization
        if self.training:
            for b in range(B):
                # Random 90 deg rotations
                rot = torch.randint(0, 4, (1,)).item()
                if rot == 1:
                    col[b], row[b] = row[b].clone(), col[b].clone()
                    col[b] = col[b].max() - col[b]
                elif rot == 2:
                    col[b] = col[b].max() - col[b]
                    row[b] = row[b].max() - row[b]
                elif rot == 3:
                    col[b], row[b] = row[b].clone(), col[b].clone()
                    row[b] = row[b].max() - row[b]
                    
                # Random flips
                if torch.rand(1).item() > 0.5:
                    col[b] = col[b].max() - col[b]
                if torch.rand(1).item() > 0.5:
                    row[b] = row[b].max() - row[b]
                
                # Random translation
                col[b] += torch.randint(0, 32, (1,), device=col.device).item()
                row[b] += torch.randint(0, 32, (1,), device=row.device).item()

        # Calculate bounding box needed
        max_col = col.max().item()
        max_row = row.max().item()

        stride = 2 ** self.depth
        min_size = stride * 2
        
        alloc_w = max(max_col + 1, min_size)
        alloc_h = max(max_row + 1, min_size)
        
        alloc_w = ((alloc_w + stride - 1) // stride) * stride
        alloc_h = ((alloc_h + stride - 1) // stride) * stride

        b_indices = torch.arange(B, device=device).view(B, 1).expand(B, N_tiles)
        
        # Build coordinates (b, y, x) for spconv
        indices = torch.stack([b_indices.flatten(), row.flatten(), col.flatten()], dim=1).int()
        
        # Deduplicate indices (if overlap exists) to satisfy spconv
        unique_indices, inverse_indices = torch.unique(indices, dim=0, return_inverse=True)
        
        return unique_indices, inverse_indices, alloc_h, alloc_w


    def loss(self, 
        annotation: torch.Tensor = None, seg_logits: torch.Tensor = None, 
        label: torch.Tensor = None, lbl_logits: torch.Tensor=None, 
        **kwargs) -> dict:

        if seg_logits is None: seg_logits = self.output.get('ann_logits')
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')

        results = {}
        if seg_logits is not None and annotation is not None:
            # logits: (B, N, C), annotation: (B, N, C)
            B = seg_logits.size(0)  # (B, N, C)
    
            total_tile = 0.0
            total_dice = 0.0
            for b in range(B):
                pred_nc = seg_logits[b]         # (N, C)
                ann_nc = annotation[b]          # (N, C)
    
                # Per-tile label loss
                total_tile = total_tile + self.ann_loss_fn(pred_nc, ann_nc)
                # Dice loss: transpose to (N, C) for per-class across N tiles
                total_dice = total_dice + self.dice_fn(pred_nc.t(), ann_nc.t())
    
            tile_loss = total_tile / B
            dice_loss = total_dice / B
            ann_loss = self.ann_weight * tile_loss + self.dice_weight * dice_loss
    
            results.update(annotation={'ann_loss': ann_loss, 'tile_loss': tile_loss, 'dice_loss': dice_loss})
            
        if lbl_logits is not None and label is not None:
            results['label'] = {'lbl_loss': self.lbl_loss_fn(lbl_logits, label)}
        return results
        
        
    @torch.no_grad()
    def predict(self, ann_logits: torch.Tensor = None, lbl_logits: torch.Tensor = None, **kwargs) -> dict:
        if ann_logits is None: ann_logits = self.output.get('ann_logits')
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')
        
        # (B, N, C)
        self.result = {}
        if ann_logits is not None:
            pred_ann = apply_pred(self.ann_mode, ann_logits)
            self.result['pred_ann'] = to_cpu(pred_ann)
        if lbl_logits is not None:
            pred_lbl = apply_pred(self.lbl_mode, lbl_logits)
            self.result['pred_lbl'] = to_cpu(pred_lbl)
        return self.result


    def metric(self, 
        annotation: torch.Tensor = None, pred_ann: torch.Tensor = None, 
        label: torch.Tensor = None, pred_lbl: torch.Tensor = None,
        threshold: float = 0.5, **kwargs):
        
        if pred_ann is None: pred_ann = self.result.get('pred_ann')
        if pred_lbl is None: pred_lbl = self.result.get('pred_lbl')
        
        result = {}
        if annotation is not None and pred_ann is not None:
            annotation = torch.as_tensor(annotation).float()
            result.update(annotation=self.ann_metric(pred_ann, annotation, threshold))
            
        if label is not None and pred_lbl is not None:
            label = torch.as_tensor(label).float()
            result.update(label=self.lbl_metric(pred_lbl, label, threshold))
        return result

