"""Simple MLP classifier for standalone tile features.

Consumes ``tile_features_collector`` output (standalone mode):
    - features: (B, F) — single feature vector per tile
    - label:    (B, C) — C-class soft/hard label

Produces:
    - logits: (B, C) — per-class logits
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Literal


def squeeze_outer_batch(features: torch.Tensor):
    """(G, B, F) -> (G*B, F)"""
    leading_shape = None
    if features.ndim == 3:
        leading_shape = features.shape[:2]          # (G, B)
        features = features.reshape(-1, features.size(-1))  # (G*B, F)
    return features, leading_shape

def unsqueeze_outer_batch(pred: torch.Tensor, leading_shape):
    """(G*B, C) -> (G, B, C)"""
    if leading_shape is not None:
        pred = pred.reshape(*leading_shape, -1)  # (G, B, C)
    return pred


class MLPClassifier(nn.Module):
    """MLP classifier for individual tile feature vectors.

    A lightweight feed-forward network that maps a single tile feature
    vector to class logits. Supports standard mini-batch training (B > 1).

    Args:
        in_features (int): Input feature dimension F from the backbone.
        n_classes (int): Number of output classes C.
        hidden_dim (int): Hidden layer dimension. Default 256.
        n_layers (int): Number of hidden layers. Default 2.
        dropout (float): Dropout rate between layers. Default 0.25.

    Input:
        features (torch.Tensor): (B, F) tile feature vectors.

    Output:
        dict with:
            - ``logits`` (torch.Tensor): (B, C) class logits.
    """

    def __init__(
        self,
        in_features: int,
        lbl_n_classes: int = 0,
        ann_n_classes: int = 0,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.25,
        mode: Literal['slide', 'mode'] = None
    ):
        super().__init__()

        layers = []
        dim_in = in_features
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(dim_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim_in = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.mode = 'slide' if not mode else mode
        if mode == 'slide':
            n_classes = ann_n_classes
        else:
            n_classes = lbl_n_classes
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, features: torch.Tensor, **kwargs) -> dict:
        """Forward pass.

        Args:
            features (torch.Tensor): (B, F) tile features.

        Returns:
            dict: ``logits`` (B, C).
        """
        features, leading_shape = squeeze_outer_batch(features)
        h = self.encoder(features)   # (B, hidden_dim)
        logits = self.head(h)        # (B, C)
        if leading_shape is not None:
            return {"ann_logits": unsqueeze_outer_batch(logits, leading_shape)}
        else:
            return {"lbl_logits": logits}
            
    def get_model_info(self):
        info_list = []
        if self.mode == 'slide': info_list.append('SEG')
        if self.mode == 'tile': info_list.append('CLS')
        return {'modality': info_list}
