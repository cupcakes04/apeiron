import torch
import torch.nn as nn
import torch.nn.functional as F
from ..helper import GatedAttention, build_loss

class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning for slide classification.

    Projects N tile features through a shared encoder, computes per-class
    gated attention weights, and aggregates into slide-level logits.
    Batch size is strictly 1 if variable N per slide.

    Args:
        in_features (int): Input feature dimension F from the backbone.
        lbl_n_classes (int): Number of output classes C.
        embed_dim (int): Projection dimension. Default 256.
        attn_dim (int): Attention hidden dimension. Default 128.
        dropout (float): Dropout rate in the projection layers. Default 0.25.

    Input:
        features (torch.Tensor): (B, N, F) tile feature vectors.

    Output:
        dict with:
            - ``logits``    (torch.Tensor): (B, C) slide-level class logits.
            - ``attention`` (torch.Tensor): (B, C, N) normalised attention weights.
    """
    def __init__(
        self,
        in_features: int,
        lbl_n_classes: int,
        embed_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.25,
        lbl_loss_type: str = 'hard_ce', 
        lbl_cls_weights: dict = None, 
        **kwargs
    ):
        super().__init__()
        self.lbl_n_classes = lbl_n_classes
        self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)

        # 1. Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Unified Gated Attention
        # This produces ONE attention score per tile, shared by all classes.
        self.attention = GatedAttention(embed_dim, attn_dim)

        # 3. Final Classifier
        # Now we only need ONE linear layer because we have ONE pooled vector.
        self.classifier = nn.Linear(embed_dim, lbl_n_classes)

    def forward(self, features: torch.Tensor, **kwargs) -> dict:
        """Forward pass.
        Args:
            features (torch.Tensor): (B, N, F) tile features.

        Returns:
            dict: ``logits`` (B, C) and ``attention`` (B, N).
        """
        # Step 1: Project Features -> (B, N, D)
        h = self.encoder(features)
        
        # Step 2: Compute Single Attention Map (Reshape to (B, 1, N) for pooling)
        a_weights = self.attention(h).permute(0, 2, 1)    # (B, 1, N)

        # Step 3: Global Pooling
        # (B, 1, N) @ (B, N, D) -> (B, 1, D)
        h_pooled = torch.bmm(a_weights, h)
        h_pooled = h_pooled.squeeze(1)           # (B, D)

        # Step 4: Classification -> (B, C)
        logits = self.classifier(h_pooled)
        
        return {"lbl_logits": logits, "attention": a_weights}
    
    def loss(self, lbl_logits: torch.Tensor, label: torch.Tensor, **kwargs) -> dict:
        lbl_loss = self.lbl_loss_fn(lbl_logits, label)
        return {'lbl_loss': lbl_loss}


