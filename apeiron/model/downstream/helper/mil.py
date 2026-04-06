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
        a_v = torch.tanh(self.attention_V(h))     # (B, N, attn_dim)
        a_u = torch.sigmoid(self.attention_U(h))  # (B, N, attn_dim)
        a_raw = self.attention_W(a_v * a_u)       # (B, N, 1)
        return F.softmax(a_raw, dim=1)            # (B, N, 1)

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