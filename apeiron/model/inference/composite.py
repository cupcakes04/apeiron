import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, List
from apeiron.utils import convert_to_list
from .utility import choose_inferencer, ModelData, LOSS_MODALITIES
from ..downstream import *

"""Composite downstream models that merge two task heads behind a shared encoder.
Architecture::

    features (1, N, F)
         │
    ┌────▼────┐
    │ encoder │  shared 2-layer MLP: F → D
    └────┬────┘
         │ h (1, N, D)
         ├──────────────────┐
    ┌────▼────┐        ┌────▼────┐
    │ head_a  │        │ head_b  │
    └─────────┘        └─────────┘

Output dicts from both heads are merged. Key prefixes avoid collisions.
"""


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

class MultiHeadModel(nn.Module):
    
    """
    A dynamic multi-head model using a shared residual encoder.
    
    Args:
        heads_dict (dict): A dictionary where keys are task names (str) 
                           and values are the head modules (nn.Module).
    """
    def __init__(self, 
        in_features, mode=None, inf_models: list|str = 'abmil',
        lbl_n_classes=0, lbl_loss_type=None, lbl_cls_weights: dict = None,
        ann_n_classes=0, ann_loss_type=None, ann_cls_weights: dict = None):
        super().__init__()

        # 1. Use our 1536-D Residual Encoder from before
        self.encoder = SharedEncoder(in_features=in_features)
        
        # 2. Register heads as a ModuleDict so PyTorch can see their weights
        heads_dict = {}
        for inf_model in convert_to_list(inf_models):
            heads_dict[inf_model] = choose_inferencer(
                in_features, mode=mode, inf_model=inf_model,
                lbl_n_classes=lbl_n_classes, lbl_loss_type=lbl_loss_type, lbl_cls_weights=lbl_cls_weights,
                ann_n_classes=ann_n_classes, ann_loss_type=ann_loss_type, ann_cls_weights=ann_cls_weights,
            )
        self.heads = nn.ModuleDict(heads_dict)

        # Initialize AWL with the specific keys we expect
        self.losser = AutomaticWeightedLoss(task_keys=LOSS_MODALITIES)

    def forward(self, features, **kwargs) -> ModelData:
        """
        Args:
            head_outputs: Dict containing sub-model outputs:
                {
                   'model_A': {'seg_logits': (B, C_seg, N), 'attention': (B, C_lbl, N), instances: (B, N, D)},
                   'model_B': {'seg_logits': (B, C_seg, N), 'attention': (B, C_lbl, N), instances: (B, N, D)}
                }
        """
        # Run Model Heads
        h = self.encoder(features)  # (1536 -> 1536)
        head_outputs = {name: head_module(h, **kwargs) for name, head_module in self.heads.items()}
            
        # Unify Heads
        mdata = ModelData(model_names=list(head_outputs.keys()))
        for out in head_outputs.values():
            mdata.assign(mode='head', **out)
        return mdata

    def loss(self, mdata: ModelData, **target) -> ModelData:
        task_losses = {}

        # --- 1. Collect Raw Losses ---
        for head in self.heads.values():
            out = head.loss(**target)
            mdata.assign(mode='loss', **out)
            flat_out = {k: v for o in out.values() for k, v in o.items()}
            for key in LOSS_MODALITIES:
                if key in flat_out:
                    task_losses[key] = flat_out[key]

        # --- 2. Normalize and Weight ---
        # We pass the dict to AWL which handles the missing keys dynamically
        if task_losses:
            mdata.composite_loss(self.losser(task_losses), **self.losser.get_importance())
        return mdata

    def predict(self, mdata: ModelData):
        for head in self.heads.values():
            res = head.predict()
            mdata.assign(mode='pred', **res)
        return mdata

    def metric(self, mdata: ModelData, threshold=0.5, **target):
        for head in self.heads.values():
            met = head.metric(threshold=threshold, **target)
            mdata.assign(mode='metric', **met)
        return mdata


    # |-----------------------------------------------|
    # |-------------------- Extra --------------------|
    # |-----------------------------------------------|

    def clear_cache(self):
        for name, head_module in self.heads.items():
            if name == 'contrastive':
                head_module.clear_cache()

    def get_contrastive_embeddings(self, features=None, text=None):
        """
        Args:
        - features -> (B, N, F)
        - text -> list[str]
        
        output: (dict)
        - img_emb -> (B,H)
        - wrd_emb -> (B,H)
        
        usage:
        ```python
        # Calculate scores (matching score for each text)
        # (4, H) @ (H, 1) -> (4, 1)
        scores = wrd_emb @ img_emb.t()

        # Calculate scores (matching score for each emb)
        # (10, H) @ (H, 1) -> (10, 1)
        scores = img_emb @ wrd_emb.t()
        ```
        """
        vectors = {}
        for name, head_module in self.heads.items():
            if name == 'contrastive':
                if features is not None:
                    features = torch.tensor(features, device=next(self.parameters()).device)
                    vectors.update(head_module.get_img_emb(self.encoder(features)))
                if text is not None:
                    vectors.update(head_module.get_wrd_emb(text))
        return vectors
        

        
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, task_keys: list):
        super().__init__()
        self.keys = task_keys
        # Initialize log_var at 0 (Weight = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(len(task_keys)))

    def forward(self, loss_dict: dict):
        # Initialize total_loss on the correct device
        device = next(iter(loss_dict.values())).device
        total_loss = torch.zeros(1, device=device)

        # 1. Count how many tasks are actually present in this batch
        num_tasks = len([k for k in self.keys if k in loss_dict])

        for i, key in enumerate(self.keys):
            if key not in loss_dict:
                continue
                
            loss = loss_dict[key]
            # 2. AUTO-BYPASS: If only 1 task is present, use standard weight (1.0)
            # This prevents the log_var from "cheating" when it's alone.
            if num_tasks <= 1:
                weighted_loss = loss
            else:
                s = self.log_vars[i]
                # Standard AWL formula
                weighted_loss = torch.exp(-s) * loss + s
            
            total_loss += weighted_loss
            
        return total_loss

    def get_importance(self):
        """
        - Weight = exp(-log_var). 
        - High log_var (uncertainty) = Low weight.
        """
        with torch.no_grad():
            # Correctly mapping keys to the learned weights
            return {
                f"{key}_weight": torch.exp(-self.log_vars[i]).item() 
                for i, key in enumerate(self.keys)
            }