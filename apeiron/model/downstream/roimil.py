import torch
import torch.nn as nn
import numpy as np
from apeiron.utils import to_cpu
from .helper import GatedAttention, build_loss, check_mode, apply_pred, get_metric_mode, LabelMetrics

class ROIMIL(nn.Module):
    """
    Dual-branch ROI-guided MIL.
    Computes a slide-level prediction from all tiles, and an ROI-level prediction
    from specific subset tiles provided via `objects`.
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

        # ROI specific
        ann_n_classes: int = None,
        ann_loss_type: str = 'hard_ce',
        ann_cls_weights: dict = None,
        **kwargs
    ):
        super().__init__()
        roi_loss_type = ann_loss_type
        roi_cls_weights = ann_cls_weights
        self.roi_n_classes = ann_n_classes or lbl_n_classes
        
        # Shared Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 1. Slide-level Branch
        self.lbl_mode = check_mode(lbl_loss_type)
        self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights)
        self.lbl_metric = LabelMetrics(mode=get_metric_mode(lbl_loss_type))
        self.slide_attention = GatedAttention(embed_dim, attn_dim)
        self.slide_classifier = nn.Linear(embed_dim, lbl_n_classes)

        # 2. ROI-level Branch
        self.roi_mode = check_mode(roi_loss_type)
        self.roi_loss_fn = build_loss(roi_loss_type, cls_weights=roi_cls_weights)
        self.roi_metric = LabelMetrics(mode=get_metric_mode(roi_loss_type))
        self.roi_attention = GatedAttention(embed_dim, attn_dim)
        self.roi_classifier = nn.Linear(embed_dim, self.roi_n_classes)

        # Dual-branch Late Fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output = {}
        self.result = {}

    def forward(self, features: torch.Tensor, objects: list = None, **kwargs) -> dict:
        # Step 1: Shared encoding -> (B, N, D)
        h = self.encoder(features)
        B, N, D = h.shape
        
        self.output = {}

        # Step 2: ROI-level MIL & Hierarchical Slide-level MIL
        if objects is None:
    
            # Fallback if no objects passed at all (Standard Tile-level MIL)
            a_slide = self.slide_attention(h).permute(0, 2, 1)               # (B, 1, N)
            h_slide = torch.bmm(a_slide, h).squeeze(1)                       # (B, D)
            lbl_logits = self.slide_classifier(h_slide)                      # (B, C_lbl)
            
            self.output["lbl_logits"] = lbl_logits
            self.output["attention"] = a_slide

        else:
            roi_logits_list = []
            roi_masks_list = []
            
            # Branch 1: Main Stream (Standard Slide-level MIL over ALL raw tiles)
            a_slide = self.slide_attention(h).permute(0, 2, 1) # (B, 1, N)
            h_slide_pool = torch.bmm(a_slide, h).squeeze(1)    # (B, D)
            
            # Branch 2: Assistance Stream (ROI-level MIL)
            roi_global_list = []

            for b in range(B):
                b_roi_logits = []
                b_roi_masks = []
                b_roi_pools = []
                
                if objects and objects[b]:
                    for roi in objects[b]:
                        ids = roi['ids']
                        if len(ids) == 0: continue
                        
                        # Extract ROI tiles -> (1, N_roi, D)
                        h_roi = h[b, ids, :].unsqueeze(0)
                        
                        # Apply ROI MIL -> (1, D)
                        a_roi = self.roi_attention(h_roi).permute(0, 2, 1)
                        h_roi_pool = torch.bmm(a_roi, h_roi).squeeze(1) # (1, D)
                        
                        b_roi_logits.append(self.roi_classifier(h_roi_pool)) # (1, C_ann)
                        b_roi_pools.append(h_roi_pool) # (1, D)
                        
                        # Create boolean mask for this ROI
                        mask = torch.zeros(N, dtype=torch.bool, device=h.device)
                        mask[ids] = True
                        b_roi_masks.append(mask)
                
                # Aggregate all ROIs in this slide to form a single global ROI assistance token
                if b_roi_pools:
                    # Mean pooling the rich ROI features -> (1, D)
                    h_roi_global = torch.mean(torch.cat(b_roi_pools, dim=0), dim=0, keepdim=True)
                else:
                    # Fallback if no ROIs in this specific slide -> zero vector (1, D)
                    h_roi_global = torch.zeros(1, D, device=h.device)
                    
                roi_global_list.append(h_roi_global)
                
                # Stack ROI logits and masks for this batch item
                if b_roi_logits:
                    roi_logits_list.append(torch.cat(b_roi_logits, dim=0))   # (Q, C_ann)
                    roi_masks_list.append(torch.stack(b_roi_masks, dim=0))   # (Q, N)
            
            # Step 3: Late Fusion & Global Slide-level Classification
            # Stack the global ROI assistance tokens -> (B, D)
            h_roi_global_batch = torch.cat(roi_global_list, dim=0)
            
            # Concatenate Main Stream with Assistance Stream -> (B, 2D)
            h_fused = torch.cat([h_slide_pool, h_roi_global_batch], dim=-1)
            
            # Compress and Classify
            h_final = self.global_fusion(h_fused) # (B, D)
            slide_logits = self.slide_classifier(h_final) # (B, C_lbl)
                
            # Store Outputs
            self.output["obj_classes"] = roi_logits_list
            self.output["obj_masks"] = roi_masks_list
            self.output["lbl_logits"] = slide_logits
            self.output["attention"] = a_slide

            
        return self.output

    def loss(self, label: torch.Tensor = None, objects: list = None, **kwargs) -> dict:
        has_slide_gt = label is not None
        has_roi_gt = objects is not None and any(len(o) > 0 for o in objects if o)

        # The core constraint
        if not has_slide_gt and not has_roi_gt:
            raise ValueError("ROIMIL requires at least one of `label` or `objects` for supervision.")

        result = {}
        
        # 1. Slide-level Loss
        if has_slide_gt:
            lbl_logits = self.output['lbl_logits']
            result['label'] = {'lbl_loss': self.lbl_loss_fn(lbl_logits, label)}
            
        # 2. ROI-level Loss
        if has_roi_gt and 'obj_classes' in self.output:
            roi_logits_list = self.output['obj_classes']
            roi_losses = []
            
            for b in range(len(objects)):
                if not objects[b] or roi_logits_list[b] is None:
                    continue
                
                # Extract ground truth ROI labels -> (Q, C)
                roi_targets = torch.tensor(
                    [roi['label'] for roi in objects[b]], 
                    device=roi_logits_list[b].device, 
                    dtype=torch.float
                )
                roi_losses.append(self.roi_loss_fn(roi_logits_list[b], roi_targets))
            
            if roi_losses:
                result['objects'] = {'obj_loss': torch.stack(roi_losses).mean()}

        return result

    @torch.no_grad()
    def predict(self, attention: torch.Tensor = None, lbl_logits: torch.Tensor = None, obj_classes: list = None, obj_masks: list = None, **kwargs) -> dict:
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')
        if attention is None: attention = self.output.get('attention')
        if obj_classes is None: obj_classes = self.output.get('obj_classes')
        if obj_masks is None: obj_masks = self.output.get('obj_masks')

        if lbl_logits is not None:
            pred_lbl = apply_pred(self.lbl_mode, lbl_logits)
            self.result['pred_lbl'] = to_cpu(pred_lbl)

        if attention is not None:
            self.result['pred_atn'] = to_cpu(attention)

        if obj_classes is not None and obj_masks is not None:
            pred_obj = []
            for b in range(len(obj_classes)):
                b_preds = []
                if obj_classes[b] is not None and obj_masks[b] is not None:
                    # Apply prediction to ROI logits -> (Q, C)
                    roi_probs = to_cpu(apply_pred(self.roi_mode, obj_classes[b]))
                    b_masks = to_cpu(obj_masks[b]).numpy()
                    for i in range(roi_probs.shape[0]):
                        # Format as expected by API contract
                        roi_dict = {
                            "ids": np.where(b_masks[i])[0].tolist(), 
                            "labels": roi_probs[i],
                            "scores": float(roi_probs[i].max())
                        }
                        b_preds.append(roi_dict)
                pred_obj.append(b_preds)
            self.result['pred_obj'] = pred_obj if len(pred_obj) > 0 else None

        return self.result

    def metric(self, label: torch.Tensor = None, objects: list = None, pred_lbl: torch.Tensor = None, pred_obj: list = None, threshold: float = 0.5, **kwargs) -> dict:
        if pred_lbl is None: pred_lbl = self.result.get('pred_lbl')
        if pred_obj is None: pred_obj = self.result.get('pred_obj')
        
        result = {}
        
        # 1. Slide-level metrics
        if label is not None and pred_lbl is not None:
            label = torch.as_tensor(label).float()
            result['label'] = self.lbl_metric(pred_lbl, label, threshold)
        
        # 2. ROI-level metrics (flattens all ROIs across the batch to compute one pooled metric)
        if objects is not None and pred_obj is not None:
            all_pred = []
            all_true = []
            for b in range(len(objects)):
                if objects[b] and b < len(pred_obj) and pred_obj[b]:
                    for i, roi in enumerate(objects[b]):
                        if i < len(pred_obj[b]):
                            all_pred.append(pred_obj[b][i]['labels'])
                            all_true.append(roi['label'])
            
            if all_pred and all_true:
                all_pred_t = torch.tensor(np.stack(all_pred))
                all_true_t = torch.tensor(np.stack(all_true)).float()
                result['objects'] = self.roi_metric(all_pred_t, all_true_t, threshold)
                
        return result