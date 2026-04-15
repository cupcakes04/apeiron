
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import build_loss, check_mode, get_metric_mode, AnnotationMetrics, LabelMetrics, apply_pred
from apeiron.utils import to_cpu


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> GELU -> Conv -> BN -> GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder block: MaxPool -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Concat skip -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """UNet for sparse tile-level segmentation with slide-level classification.

    Rasterises N tile features onto a 2-D grid, applies a UNet
    encoder-decoder, then reads out per-pixel class logits at every
    non-empty output position. Supports B>=1 (each sample rasterised independently).

    A slide-level classification head uses the segmentation logits as
    attention weights (softmax over N tiles) to pool bottleneck features
    back to a single slide-level vector per annotation class, then maps
    to ``lbl_n_classes`` logits — effectively UNet + ABMIL in one model.

    Args:
        in_features (int): Input feature dimension F.
        ann_n_classes (int): Number of segmentation classes C_ann.
        lbl_n_classes (int): Number of slide-level label classes C_lbl.
            If 0 or ``None``, no classification head is built. Default 0.
        base_channels (int): Channel width of the first encoder stage. Default 64.
        depth (int): Number of encoder downsampling stages. Default 3.
        tile_size (int): Physical size of each tile in coordinate space.
            Grid dimensions are computed as coord_range / tile_size.
            E.g., tile_size=224 with coords spanning 75k×100k → ~335×446 grid. Default 224.

    Input:
        features (torch.Tensor): (B, N, F) tile features.
        coords   (torch.Tensor): (B, N, 2) tile spatial coordinates.

    Output:
        dict with:
            - ``ann_logits``  (torch.Tensor): (B, C_ann, N) per-class logits at original positions.
            - ``attention``   (torch.Tensor): (B, C_ann, N) normalised attention weights.
            - ``lbl_logits``  (torch.Tensor): (B, C_lbl) slide-level logits (if lbl_n_classes > 0).
    """

    def __init__(
        self,
        in_features: int,
        ann_n_classes: int = 0,
        lbl_n_classes: int = 0,
        base_channels: int = 256,
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
        
        self.loss_fn = build_loss(ann_loss_type, cls_weights=ann_cls_weights, **kwargs)
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
        self.enc_first = ConvBlock(base_channels, base_channels)
        self.encoders = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.encoders.append(DownBlock(ch, ch * 2))
            ch *= 2
        self.bottleneck_ch = ch  # deepest channel width

        # Decoder
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.decoders.append(UpBlock(ch, ch // 2))
            ch //= 2

        # Segmentation head
        if self.ann_n_classes >= 0:
            self.seg_head = nn.Conv2d(ch, ann_n_classes, kernel_size=1)

        # Classification head
        if self.lbl_n_classes >= 0:
            self.attention_V = nn.Sequential(
                nn.Conv2d(self.bottleneck_ch, base_channels, kernel_size=1),
                nn.Tanh(),
                nn.Dropout(0.1)
            )
            self.attention_U = nn.Sequential(
                nn.Conv2d(self.bottleneck_ch, base_channels, kernel_size=1),
                nn.Sigmoid(),
                nn.Dropout(0.1)
            )
            self.attention_weights = nn.Conv2d(base_channels, 1, kernel_size=1)
            self.classifier = nn.Linear(self.bottleneck_ch, self.lbl_n_classes)
            self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)
            self.lbl_metric = LabelMetrics(mode=get_metric_mode(lbl_loss_type))
            self.lbl_mode = check_mode(lbl_loss_type)

        self.output = {}
        self.result = {}

    def forward(self, features: torch.Tensor, coords: torch.Tensor, coords_size: int, **kwargs) -> dict:
        """Forward pass processing the entire batch.

        Args:
            features (torch.Tensor): (B, N, F) tile features.
            coords (torch.Tensor): (B, N, 2) tile coordinates.
            coords_size (int): Size of one tile in coordinate space.

        Returns:
            dict: ``seg_logits`` (B, N, C_ann), ``attention`` (B, N), ``lbl_logits`` (B, C)
        """
        B, N_tiles, _ = features.shape

        # 1. Project features
        feats = self.input_proj(features)  # (B, N, base_channels)
        grid, (row, col, stride) = self._compute_adaptive_grid(coords, feats, coords_size)

        # 4. UNet encoder
        skips = []
        x_enc = self.enc_first(grid)
        skips.append(x_enc)
        for enc in self.encoders:
            x_enc = enc(x_enc)
            skips.append(x_enc)

        bottleneck = x_enc  # (B, bottleneck_ch, H', W')

        # seg_map is (B, C, H, W), row/col are (B, N)
        device = x_enc.device
        b_indices = torch.arange(B, device=device).view(B, 1).expand(B, N_tiles)

        if self.ann_n_classes >= 0:
            # 5. UNet decoder
            skips.pop()  # bottleneck skip not used
            for dec in self.decoders:
                x_enc = dec(x_enc, skips.pop())

            seg_map = self.seg_head(x_enc)  # (B, C_ann, H, W)

            # 6. Sample back at original N input positions
            seg_logits = seg_map[b_indices, :, row, col].transpose(1, 2)  # (B, C_ann, N)
            result = {'ann_logits': seg_logits.transpose(1, 2)} # seg_logits: (B, N, C_ann)

        # 7. Classification head via ABMIL on bottleneck features
        if self.lbl_n_classes >= 0:
            bn_row = row // stride
            bn_col = col // stride
            
            B_bn, C_bn, H_bn, W_bn = bottleneck.shape
            
            # Mask of valid bottleneck cells
            bn_mask = torch.zeros((B_bn, 1, H_bn, W_bn), device=device, dtype=torch.bool)
            for b in range(B_bn):
                bn_mask[b, 0, bn_row[b], bn_col[b]] = True
                
            A_V = self.attention_V(bottleneck)  # (B, 256, H_bn, W_bn)
            A_U = self.attention_U(bottleneck)  # (B, 256, H_bn, W_bn)
            A_raw = self.attention_weights(A_V * A_U)  # (B, 1, H_bn, W_bn)
            
            # Masked softmax
            A_raw = A_raw.masked_fill(~bn_mask, -1e9)
            A_flat = A_raw.view(B_bn, 1, -1)
            A_flat = torch.softmax(A_flat, dim=-1)
            A_grid = A_flat.view(B_bn, 1, H_bn, W_bn)  # (B, 1, H_bn, W_bn)
            
            # Pool features
            M = (A_grid * bottleneck).sum(dim=(2, 3))  # (B, bottleneck_ch)
            
            lbl_logits = self.classifier(M)  # (B, C_lbl)
            result['lbl_logits'] = lbl_logits
            result['attention'] = A_grid.squeeze(1)  # (B, H_bn, W_bn)
            
        self.output = result
        return self.output


    def _compute_adaptive_grid(self, coords, feats, coords_size):
        
        B, N_tiles, C_feat = feats.shape

        # 2. Compute adaptive grid dimensions from coordinate range across the batch
        x, y = coords[:, :, 0].float(), coords[:, :, 1].float()
        
        # Per-sample range
        x_mins, x_maxs = x.min(dim=1).values, x.max(dim=1).values
        y_mins, y_maxs = y.min(dim=1).values, y.max(dim=1).values
        
        x_range = (x_maxs - x_mins).max().clamp(min=coords_size)
        y_range = (y_maxs - y_mins).max().clamp(min=coords_size)
        
        grid_w = int((x_range / coords_size).ceil().item()) + 1
        grid_h = int((y_range / coords_size).ceil().item()) + 1

        # Calculate allocation size to prevent BatchNorm crashes in 1x1 bottlenecks
        # and to ensure perfect divisibility through UNet pooling layers.
        stride = 2 ** self.depth
        min_size = stride * 2  # min 16x16 for depth=3
        
        alloc_w = max(grid_w, min_size)
        alloc_h = max(grid_h, min_size)
        
        # Round up to nearest multiple of stride
        alloc_w = ((alloc_w + stride - 1) // stride) * stride
        alloc_h = ((alloc_h + stride - 1) // stride) * stride

        # 3. Rasterise onto adaptive grid
        # _coords_to_grid_indices needs flat or handles batched? It flattens. Let's do it per sample manually.
        # Actually, let's vectorise:
        x_mins_b = x_mins.view(B, 1)
        y_mins_b = y_mins.view(B, 1)
        
        col = ((x - x_mins_b) / x_range * (alloc_w - 1)).round().long().clamp(0, alloc_w - 1)
        row = ((y - y_mins_b) / y_range * (alloc_h - 1)).round().long().clamp(0, alloc_h - 1)

        grid = feats.new_zeros(B, C_feat, alloc_h, alloc_w)
        count = feats.new_zeros(B, 1, alloc_h, alloc_w)

        # Vectorised scatter-add across batch
        # We need to flatten the batch to use index_put_ or scatter, or just use a simple loop over B 
        # since B is usually small or N is fixed. Let's do a fast loop over B for scatter:
        for b in range(B):
            grid[b, :, row[b], col[b]] += feats[b].t()
            count[b, :, row[b], col[b]] += 1
            
        count = count.clamp(min=1)
        grid = grid / count  # (B, C_feat, H, W)
        return grid, (row, col, stride)

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
                total_tile = total_tile + self.loss_fn(pred_nc, ann_nc)
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