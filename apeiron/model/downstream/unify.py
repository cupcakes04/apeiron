import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *

# DOWNSTREAM_NAMES = {
#     'MLPClassifier': "mlp",
#     'ABMIL': "abmil",
#     'SparseUNet': "unet",
#     'SparseDETR': "detr",
#     'GenerativeVLM': "genvl",
#     'ContrastiveVLM': "convl",
# }

# DOWNSTREAM_TYPES = {
#     'mlp': "CLS",
#     'abmil': "MIL",
#     'unet': "SEG",
#     'detr': "DET",
#     'text': "VLM",
# }

# MODALITIES = ['label', 'annotation', 'objects', 'text']


# def make_linear_layer(n_classes: int, num_models: int = 2) -> nn.Sequential:
#     """
#     Returns a refinement head:
#     Input: (B, n_classes * num_models)
#     Output: (B, n_classes)
#     """
#     return nn.Sequential(
#         nn.Linear(n_classes * num_models, n_classes * num_models),
#         nn.GELU(),
#         nn.Linear(n_classes * num_models, n_classes)
#     )


# ============================================================================
# Unification by Shared encoder
# ============================================================================

class SharedEncoder(nn.Module):
    """Two-layer MLP feature projection shared across heads.

    Args:
        in_features (int): Input feature dimension F.
        embed_dim (int): Output embedding dimension D. Default 256.
        dropout (float): Dropout rate. Default 0.25.

    Input:
        features (torch.Tensor): ``(1, N, F)`` or ``(B, F)`` tile features.

    Output:
        torch.Tensor: ``(1, N, D)`` or ``(B, D)`` projected features.
    """

    def __init__(self, in_features: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.Dropout(dropout),
        )
        # LayerNorm helps the sum stay numerically stable
        self.norm = nn.LayerNorm(in_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # The Residual Connection: Identity + Refinement
        residual = features 
        out = self.net(features)
        
        # Return the sum, normalized
        return self.norm(out + residual)
        

# # ============================================================================
# # Shared [MIL, CLS, SEG, DET] heads
# # ============================================================================

# class SharedMILHead(nn.Module):
#     def __init__(self, n_classes: int, num_models: int = 2, fused_dim: int = 0, base_channels: int = 256):
#         super().__init__()
#         self.n_classes = n_classes
#         self.num_models = num_models
        
#         # GATE FOR EACH TILE:
#         # Input: (B, N, C * num_models) -> Output: (B, N, num_models)
#         # This allows a unique weight per model, per tile.
#         self.tile_gate = nn.Sequential(
#             nn.Linear(n_classes * num_models, base_channels),
#             nn.GELU(),
#             nn.Linear(base_channels, num_models) 
#         )

#         self.classifiers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(fused_dim, base_channels),
#                 nn.GELU(),
#                 nn.Linear(base_channels, 1)
#             ) for _ in range(n_classes)
#         ])
        
#         self.log_temp = nn.Parameter(torch.zeros(1)) # Starts at T=1.0

#     def forward(self, all_attn, all_insts):
#         """
#         unified_raw_attn: [(B, C, N), (B, C, N)] -> Raw scores from your encoders
#         lbl_logits: [(B, C), (B, C)] -> Slide logits
#         """
        
#         # 1. Prepare Inputs
#         # stacked_logits: (B, M, C, N)
#         stacked_logits = torch.stack(all_attn, dim=1) 
#         # gate_input: (B, N, M*C) -> Every tile's scores across all models
#         gate_input = torch.cat(all_attn, dim=1).transpose(1, 2)
#         # fused_features: (B, N, D_total)
#         fused_features = torch.cat(all_insts, dim=-1)

#         # 2. PER-TILE GATE CALCULATION
#         # gate_out: (B, N, M) -> Raw "trust" scores for each model at each tile
#         gate_out = self.tile_gate(gate_input)
        
#         # Softmax over the 'M' dimension: weights for Model 0 and Model 1 sum to 1.0 per tile
#         # gate_weights: (B, M, 1, N) after transpose and unsqueeze
#         gate_weights = F.softmax(gate_out, dim=-1).transpose(1, 2).unsqueeze(2)

#         # 3. UNIFIED RAW LOGITS (The Global Scale for Grids/Slides)
#         # This merges the two models' opinions based on the per-tile gate
#         # Result shape: (B, C, N)
#         unified_raw_attn = (stacked_logits * gate_weights).sum(dim=1)

#         # 4. NORMALIZED ATTENTION (The Local Scale for Pooling)
#         # Softmax over N to get the distribution for this specific bag
#         temp = torch.exp(self.log_temp).clamp(0.1, 5.0)
#         unified_attention = F.softmax(unified_raw_attn / temp, dim=-1)

#         # 5. POOLING & CLASSIFICATION
#         # h_fused: (B, C, D_total)
#         h_fused = torch.bmm(unified_attention, fused_features)

#         mil_logits = torch.cat([
#             self.classifiers[c](h_fused[:, c, :]) for c in range(self.n_classes)
#         ], dim=-1)

#         return {'attention': unified_raw_attn, 'lbl_logits': mil_logits}

# class SharedSEGHead(nn.Module):
#     def __init__(self, n_classes: int, num_models: int = 2):
#         """
#         The input dimension is the SUM of both models' feature dimensions
#         """
#         super().__init__()
#         # Blends the spatial 'opinions' of both models
#         self.refinement = make_linear_layer(n_classes, num_models)

#     def forward(self, all_seg):
#         # all_seg: [(B, N, C), (B, N, C)]
#         combined = torch.cat(all_seg, dim=-1)
#         return {'ann_logits': self.refinement(combined)}   # (B, N, C)

# class SharedCLSHead(nn.Module):
#     def __init__(self, n_classes: int, num_models: int = 2):
#         super().__init__()
#         # Input: (B, C * num_models) -> Output: (B, C)
#         self.refinement = make_linear_layer(n_classes, num_models)

#     def forward(self, all_logits):
#         # all_logits: [(B, C), (B, C)]
#         combined = torch.cat(all_logits, dim=1) # Results in (B, 2*C)
#         return {'lbl_logits': self.refinement(combined)}   # (B, C)

# class SharedDETHead(nn.Module):
#     def __init__(self, n_classes: int, num_models: int = 2):
#         super().__init__()
#         # C+1 because of the 'no-object' class in DETR
#         n_classes += 1
#         self.class_refinement = make_linear_layer(n_classes, num_models)
#         # Input channels = num_models (one mask per model per query)
#         self.mask_refinement = nn.Conv1d(num_models, 1, kernel_size=1)

#     def forward(self, all_class_logits, all_masks):
#         # all_class_logits: list of [(B, Q, C+1), (B, Q, C+1)]
#         # all_masks: list of [(B, Q, N), (B, Q, N)]
#         class_combined = torch.cat(all_class_logits, dim=-1) # (B, Q, 2*(C+1))
#         class_refined = self.class_refinement(class_combined) # (B, Q, C+1)
        
#         # all_masks: list of [(B, Q, N), (B, Q, N)]
#         # We need to treat 'Q' as part of the batch or the channel
#         # Let's stack them so models are in the channel dim
#         mask_combined = torch.stack(all_masks, dim=2) # (B, Q, 2, N)
#         B, Q, M, N = mask_combined.shape
        
#         # Reshape to (B*Q, M, N) to use Conv1d as a blender
#         mask_combined = mask_combined.view(-1, M, N) 
#         mask_refined = self.mask_refinement(mask_combined) # (B*Q, 1, N)
        
#         return {'obj_classes': class_refined, 'obj_masks': mask_refined.view(B, Q, N)}