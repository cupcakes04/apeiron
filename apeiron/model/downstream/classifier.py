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
from apeiron.utils import to_cpu
from .helper import GatedAttention, build_loss, check_mode, apply_pred, get_metric_mode, LabelMetrics


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
        mode: Literal['slide', 'mode'] = None,
        lbl_loss_type: str = 'hard_ce', 
        lbl_cls_weights: dict = None,
        ann_loss_type: str = 'hard_ce', 
        ann_cls_weights: dict = None,
        **kwargs
    ):
        super().__init__()
        
        self.lbl_n_classes = lbl_n_classes
        self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights, **kwargs)
        self.lbl_metric = LabelMetrics(mode=get_metric_mode(lbl_loss_type))
        self.lbl_mode = check_mode(lbl_loss_type)

        self.ann_n_classes = ann_n_classes
        self.ann_loss_fn = build_loss(ann_loss_type, cls_weights=ann_cls_weights, **kwargs)
        self.ann_metric = LabelMetrics(mode=get_metric_mode(ann_loss_type))
        self.ann_mode = check_mode(ann_loss_type)

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
            self.modality = ['annotation']
        else:
            n_classes = lbl_n_classes
            self.modality = ['label']
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, features: torch.Tensor, **kwargs) -> dict:
        """Forward pass.

        Args:
            features (torch.Tensor): (B, F) tile features.

        Returns:
            dict: ``logits`` (B, C).
        """
        features, leading_shape = self.squeeze_outer_batch(features)
        h = self.encoder(features)   # (B, hidden_dim)
        logits = self.head(h)        # (B, C)
        if leading_shape is not None:
            self.output = {"ann_logits": self.unsqueeze_outer_batch(logits, leading_shape)}
        else:
            self.output = {"lbl_logits": logits}
        return self.output

    @staticmethod
    def squeeze_outer_batch(features: torch.Tensor):
        """(G, B, F) -> (G*B, F)"""
        leading_shape = None
        if features.ndim == 3:
            leading_shape = features.shape[:2]          # (G, B)
            features = features.reshape(-1, features.size(-1))  # (G*B, F)
        return features, leading_shape

    @staticmethod
    def unsqueeze_outer_batch(pred: torch.Tensor, leading_shape):
        """(G*B, C) -> (G, B, C)"""
        if leading_shape is not None:
            pred = pred.reshape(*leading_shape, -1)  # (G, B, C)
        return pred

            
    def loss(self, 
        lbl_logits: torch.Tensor = None, label: torch.Tensor = None, 
        ann_logits: torch.Tensor = None, annotation: torch.Tensor = None, **kwargs) -> dict:

        if ann_logits is None: ann_logits = self.output.get('ann_logits')
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')
        
        result = {}
        if annotation is not None and ann_logits is not None:
            result.update(annotation={'ann_loss': self.ann_loss_fn(ann_logits, label)})
        if label is not None and lbl_logits is not None:
            result.update(label={'lbl_loss': self.lbl_loss_fn(lbl_logits, label)})
        return result


    @torch.no_grad()
    def predict(self, ann_logits: torch.Tensor = None, lbl_logits: torch.Tensor = None, **kwargs) -> dict:
        # (B, N, C)
        if ann_logits is None: ann_logits = self.output.get('ann_logits')
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')
        
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
