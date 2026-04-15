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
    ):
        super().__init__()
        self.ann_n_classes = ann_n_classes
        self.n_queries = n_queries

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

        return {"obj_classes": obj_classes, "obj_masks": obj_masks}
        
    def get_model_info(self):
        return {'modality': ['DET']}

