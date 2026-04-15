import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List

# ==============================================================================
# 1. emb Feature Aggregation (Perceiver Resampler - Best Practice)
# ==============================================================================
class PerceiverResampler(nn.Module):
    """
    Industry standard for Vision-Language Models (used in Flamingo, LLaVA-Next).
    Compresses an arbitrary number of emb tiles (N) into a fixed number of 
    visual tokens (K) using Cross-Attention. 
    
    Two modes:
        - 'slide': input (B, N, F) -> output (B, num_latents, hidden_dim). Uses full
          cross-attention to compress N tiles into num_latents learned slots.
        - 'tile': input (B, F)    -> output (B, 1, hidden_dim). Input is already a
          single global embedding; just project it. No cross-attention needed.
    """
    def __init__(self, in_features, mode: Literal['slide', 'tile'], hidden_dim=512, num_latents=32, num_heads=8):
        super().__init__()
        self.mode = mode
        self.proj_in = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()

        if mode == 'slide':
            self.num_latents = num_latents
            # Learnable latent queries (These act as 'slots' that pull information from the emb)
            self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
            # Cross-Attention: Latents (Query) attend to emb Features (Key/Value)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.layer_norm_latents = nn.LayerNorm(hidden_dim)
            self.layer_norm_context = nn.LayerNorm(hidden_dim)
            # FFN to process the updated latents
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
            
        elif mode =='tile':
            self.num_latents = 1

    def forward(self, x):
        """
        'slide': x (B, N, F) -> (B, num_latents, hidden_dim)
        'tile' : x (B, F)    -> (B, 1,           hidden_dim)
        """
        if self.mode == 'tile':
            return self.proj_in(x).unsqueeze(1)  # (B, 1, hidden_dim)
            
        elif self.mode == 'slide':
            if x.ndim == 2:
                x = x.unsqueeze(0)

            # mode == 'tiles'
            B = x.size(0)
            context = self.proj_in(x)  # (B, N, hidden_dim)

            # Expand latents for the batch
            latents = self.latents.expand(B, -1, -1)  # (B, num_latents, hidden_dim)

            # 1. Cross Attention
            q = self.layer_norm_latents(latents)
            k = v = self.layer_norm_context(context)

            # self.cross_attn returns (attn_output, attn_weights)
            attn_out, _ = self.cross_attn(q, k, v)
            latents = latents + attn_out

            # 2. Feed Forward
            latents = latents + self.ffn(self.layer_norm_ffn(latents))
            
            return latents # (B, num_latents, hidden_dim)

