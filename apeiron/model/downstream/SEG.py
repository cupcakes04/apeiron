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

try:
    import spconv.pytorch as spconv
except ImportError:
    raise ImportError("Please install spconv to use SparseUNet: pip install spconv-cu118 (or cu121)")

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
        base_channels: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        self.ann_n_classes = ann_n_classes
        self.lbl_n_classes = lbl_n_classes
        self.depth = depth
        self.bottleneck_ch = base_channels * (2 ** depth)

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
            L = 256
            self.attention_V = nn.Sequential(
                nn.Linear(base_channels, L),
                nn.Tanh(),
                nn.Dropout(0.1)
            )
            self.attention_U = nn.Sequential(
                nn.Linear(base_channels, L),
                nn.Sigmoid(),
                nn.Dropout(0.1)
            )
            self.attention_weights = nn.Linear(L, 1)
            self.classifier = nn.Linear(base_channels, self.lbl_n_classes)


    def forward(self, features: torch.Tensor, coords: torch.Tensor, coords_size: int, **kwargs) -> dict:
        B, N_tiles, C_in = features.shape

        # 1. Project features
        feats = self.input_proj(features)

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

        b_indices = torch.arange(B, device=features.device).view(B, 1).expand(B, N_tiles)
        
        # Build coordinates (b, y, x) for spconv
        indices = torch.stack([b_indices.flatten(), row.flatten(), col.flatten()], dim=1).int()
        feats_flat = feats.view(-1, feats.size(-1))
        
        # Deduplicate indices (if overlap exists) to satisfy spconv
        unique_indices, inverse_indices = torch.unique(indices, dim=0, return_inverse=True)
        if unique_indices.size(0) < indices.size(0):
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
            
            # ABMIL Pass
            A_V = self.attention_V(feats_dense)
            A_U = self.attention_U(feats_dense)
            A_raw = self.attention_weights(A_V * A_U)
            
            A_weights = torch.softmax(A_raw, dim=1)
            
            # Pool features
            M = (A_weights * feats_dense).sum(dim=1)
            
            lbl_logits = self.classifier(M)
            result['lbl_logits'] = lbl_logits
            result['attention'] = A_weights.squeeze(-1)
            
        return result


    def get_model_info(self):
        info_list = []
        if getattr(self, 'ann_n_classes', 0) > 0: info_list.append('SEG')
        if getattr(self, 'lbl_n_classes', 0) > 0: info_list.append('MIL')
        return {'modality': info_list}


"""Sparse UNet operating on tile features treated as pixels on an irregular grid.

Consumes ``slide_features_collector`` output:
    - features: (B, N, F) — N tile features of dimension F
    - coords:   (B, N, 2) — spatial (x, y) positions of each tile
    - label:    (B, C)    — C-class soft/hard label

Each tile is treated as a single pixel. The model rasterises the sparse
tiles onto a 2-D grid, runs a standard UNet encoder-decoder, then samples
back at the original N input positions — preserving alignment with input
coordinates and annotations.

Produces:
    - seg_logits:  (B, C_ann, N) — per-class segmentation logits at the original N positions
    - lbl_logits:  (B, C_lbl)    — slide-level classification logits (attention-pooled)
    - attention:   (B, C_ann, N) — normalised attention weights over tiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SparseGridUNet(nn.Module):
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
    ):
        super().__init__()
        self.ann_n_classes = ann_n_classes
        self.lbl_n_classes = lbl_n_classes
        self.depth = depth

        # Project input features to base_channels
        self.input_proj = nn.Linear(in_features, base_channels)

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
            
            # Sample attention back to N original tile positions
            A_sampled = A_grid[b_indices, 0, bn_row, bn_col]  # (B, N)
            result['attention'] = A_sampled
            
        return result


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


    def get_model_info(self):
        info_list = []
        if self.ann_n_classes >= 0: info_list.append('SEG')
        if self.lbl_n_classes >= 0: info_list.append('MIL')
        return {'modality': info_list}
