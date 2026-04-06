import torch
import torch.nn as nn
import torch.nn.functional as F
from ..helper import BatchedGATLayer, GatedAttention, build_loss

class GATMIL(nn.Module):
    """GAT-based Multiple Instance Learning for slide classification.
    
    Encodes tile features using a Graph Attention Network (GAT) over a spatial KNN graph
    before applying Attention-Based MIL (ABMIL) pooling.
    """
    def __init__(
        self,
        in_features: int,
        lbl_n_classes: int,
        embed_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.25,
        k_neighbors: int = 32,
        num_heads: int = 4,
        lbl_loss_type: str = 'hard_ce', 
        lbl_cls_weights: dict = None, 
        **kwargs
    ):
        super().__init__()
        self.lbl_n_classes = lbl_n_classes
        self.k_neighbors = k_neighbors
        self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)
        
        # 1. Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. GAT Layer
        self.gat = BatchedGATLayer(embed_dim, embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 3. Unified Gated Attention (ABMIL)
        self.attention = GatedAttention(embed_dim, attn_dim)

        # 4. Final Classifier
        self.classifier = nn.Linear(embed_dim, lbl_n_classes)

    def _get_knn_indices(self, coords: torch.Tensor, k: int) -> torch.Tensor:
        B, N, _ = coords.shape
        actual_k = min(k + 1, N) # including self
        
        topk_indices = []
        chunk_size = 2000
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk_coords = coords[:, i:end, :]
            dist = torch.cdist(chunk_coords.float(), coords.float())
            _, topk_idx = torch.topk(dist, actual_k, dim=-1, largest=False)
            topk_indices.append(topk_idx)
            
        return torch.cat(topk_indices, dim=1)

    def forward(self, features: torch.Tensor, coords: torch.Tensor = None, **kwargs) -> dict:
        """Forward pass.
        Args:
            features (torch.Tensor): (B, N, F) tile features.
            coords (torch.Tensor, optional): (B, N, 2) tile coordinates for spatial graph.

        Returns:
            dict: ``lbl_logits`` (B, C) and ``attention`` (B, N).
        """
        # Step 1: Project Features -> (B, N, D)
        h = self.encoder(features)
        
        # Step 2: Spatial GAT encoding
        if coords is not None:
            topk_indices = self._get_knn_indices(coords, self.k_neighbors)
        else:
            # If no coords provided, fallback to sequential nearest or self
            B, N, _ = h.shape
            actual_k = min(self.k_neighbors + 1, N)
            topk_indices = torch.arange(actual_k, device=h.device).view(1, 1, actual_k).expand(B, N, actual_k)
            
        h_g = self.gat(h, topk_indices)
        
        # Residual connection
        h = h + F.gelu(h_g)

        # Step 3: Compute Single Attention Map (Reshape to (B, 1, N) for pooling)
        a_weights = self.attention(h).permute(0, 2, 1)    # (B, 1, N)

        # Step 4: Global Pooling
        h_pooled = torch.bmm(a_weights, h)
        h_pooled = h_pooled.squeeze(1)           # (B, D)

        # Step 5: Classification -> (B, C)
        logits = self.classifier(h_pooled)

        return {"lbl_logits": logits, "attention": a_weights}

    def loss(self, lbl_logits: torch.Tensor, label: torch.Tensor, **kwargs) -> dict:
        lbl_loss = self.lbl_loss_fn(lbl_logits, label)
        return {'lbl_loss': lbl_loss}