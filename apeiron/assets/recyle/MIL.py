import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    """Unified Gated Attention mechanism for Multiple Instance Learning."""
    def __init__(self, embed_dim: int, attn_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(embed_dim, attn_dim)
        self.attention_U = nn.Linear(embed_dim, attn_dim)
        self.attention_W = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): (B, N, D) pooled features.
        Returns:
            torch.Tensor: (B, 1, N) normalized attention weights.
        """
        a_v = torch.tanh(self.attention_V(h))    # (B, N, attn_dim)
        a_u = torch.sigmoid(self.attention_U(h))  # (B, N, attn_dim)
        a_raw = self.attention_W(a_v * a_u)       # (B, N, 1)
        return F.softmax(a_raw, dim=1)           # (B, 1, N)


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
    ):
        super().__init__()
        self.lbl_n_classes = lbl_n_classes

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

    def get_model_info(self):
        return {'modality': ['MIL']}


class BatchedGATLayer(nn.Module):
    """Memory-efficient batched GAT layer using KNN indices."""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        
    def forward(self, x: torch.Tensor, topk_indices: torch.Tensor):
        # x: (B, N, in_dim)
        # topk_indices: (B, N, K)
        B, N, _ = x.shape
        _, _, K = topk_indices.shape
        
        h = self.proj(x)
        h_head = h.view(B, N, self.num_heads, self.head_dim)
        
        h_head_t = h_head.transpose(1, 2) # (B, num_heads, N, head_dim)
        e_src = (h_head_t * self.attn_src).sum(dim=-1).transpose(1, 2) # (B, N, num_heads)
        e_dst = (h_head_t * self.attn_dst).sum(dim=-1).transpose(1, 2) # (B, N, num_heads)
        
        b_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        e_dst_neighbors = e_dst[b_idx, topk_indices] # (B, N, K, num_heads)
        e_src_expanded = e_src.unsqueeze(2) # (B, N, 1, num_heads)
        
        e = self.leaky_relu(e_src_expanded + e_dst_neighbors) # (B, N, K, num_heads)
        
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        
        h_neighbors = h_head[b_idx, topk_indices] # (B, N, K, num_heads, head_dim)
        
        out = (alpha.unsqueeze(-1) * h_neighbors).sum(dim=2) # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, self.out_dim)
        
        return out


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
    ):
        super().__init__()
        self.lbl_n_classes = lbl_n_classes
        self.k_neighbors = k_neighbors
        
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

    def get_model_info(self):
        return {'modality': ['MIL']}