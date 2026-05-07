import torch
import torch.nn as nn
import numpy as np
from apeiron.utils import to_cpu
from .helper import GatedAttention, build_loss, check_mode, apply_pred, get_metric_mode, LabelMetrics, AnnotationMetrics

class Mask2FormerMIL(nn.Module):
    """
    Semantic Mask2Former architecture adapted for WSI Multiple Instance Learning.
    Handles dense, pixel-wise/tile-wise annotations (B, N, C) by using class-specific queries 
    in a Transformer Decoder to predict dense masks, and utilizes Dual-Branch Late Fusion to 
    combine the query-derived global context with a standard slide-level MIL branch.
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
        
        # Annotation specific
        ann_n_classes: int = None,
        ann_weight: float = 1.0,
        dice_weight: float = 1.0,
        ann_loss_type: str = 'hard_ce',
        ann_cls_weights: dict = None,
        
        # Transformer specific
        num_heads: int = 4,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        self.lbl_n_classes = lbl_n_classes
        self.ann_n_classes = ann_n_classes or lbl_n_classes
        
        # 1. Shared Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. Pixel Decoder (Contextualize independent tiles)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True
        )
        self.pixel_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Transformer Decoder (Class-specific queries)
        # Each query is responsible for finding a specific semantic class across the entire slide
        self.queries = nn.Parameter(torch.randn(1, self.ann_n_classes, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Dense Mask Prediction Projections
        self.mask_proj = nn.Linear(embed_dim, embed_dim)
        self.pixel_proj = nn.Linear(embed_dim, embed_dim)
        
        # 5. Slide-Level Branch (Main Stream)
        self.lbl_mode = check_mode(lbl_loss_type)
        self.lbl_loss_fn = build_loss(lbl_loss_type, cls_weights=lbl_cls_weights)
        self.lbl_metric = LabelMetrics(mode=get_metric_mode(lbl_loss_type))
        
        self.slide_attention = GatedAttention(embed_dim, attn_dim)
        self.slide_classifier = nn.Linear(embed_dim, lbl_n_classes)
        
        # 6. Annotation-Level Branch (Assistance Stream)
        self.ann_mode = check_mode(ann_loss_type)
        self.ann_loss_fn = build_loss(ann_loss_type, cls_weights=ann_cls_weights, **kwargs)
        self.dice_fn = build_loss('dice', cls_weights=ann_cls_weights)
        self.ann_weight = ann_weight
        self.dice_weight = dice_weight
        self.ann_metric = AnnotationMetrics(mode=get_metric_mode(ann_loss_type))
        
        # 7. Dual-Branch Late Fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = {}
        self.result = {}
        
    def forward(self, features: torch.Tensor, annotation: torch.Tensor = None, **kwargs) -> dict:
        self.output = {}
        B, N, _ = features.shape
        
        # Step 1: Base Encoding -> (B, N, D)
        h = self.encoder(features)
        
        # Step 2: Pixel Decoder (Enrich tiles with spatial/sequence context)
        pixel_features = self.pixel_decoder(h) # (B, N, D)
        
        # Step 3: Transformer Decoder (Queries attend to pixel_features)
        q = self.queries.expand(B, -1, -1) # (B, C_ann, D)
        q_out = self.transformer_decoder(tgt=q, memory=pixel_features) # (B, C_ann, D)
        
        # Step 4: Dense Mask Prediction (Annotation stream)
        # Map queries to mask embeddings and tiles to pixel embeddings
        mask_emb = self.mask_proj(q_out) # (B, C_ann, D)
        pix_emb = self.pixel_proj(pixel_features) # (B, N, D)
        
        # Dot product between mask queries and tile embeddings -> (B, C_ann, N)
        ann_logits = torch.bmm(mask_emb, pix_emb.transpose(1, 2))
        ann_logits = ann_logits.transpose(1, 2) # (B, N, C_ann)
        
        # Step 5: Hierarchical Slide-Level Classification (Late Fusion)
        # Main Stream: Standard MIL over the context-enriched tile features
        a_slide = self.slide_attention(pixel_features).permute(0, 2, 1) # (B, 1, N)
        h_slide_pool = torch.bmm(a_slide, pixel_features).squeeze(1)    # (B, D)
        
        # Assistance Stream: Global context derived from the class-specific semantic queries
        h_ann_global = q_out.mean(dim=1) # (B, D)
        
        # Fusion
        h_fused = torch.cat([h_slide_pool, h_ann_global], dim=-1) # (B, 2D)
        h_final = self.global_fusion(h_fused) # (B, D)
        lbl_logits = self.slide_classifier(h_final) # (B, C_lbl)
        
        # Store Outputs
        self.output["lbl_logits"] = lbl_logits
        self.output["ann_logits"] = ann_logits
        self.output["attention"] = a_slide
        
        return self.output
        
    def loss(self, label: torch.Tensor = None, annotation: torch.Tensor = None, **kwargs) -> dict:
        result = {}
        
        # 1. Slide-level Loss
        if label is not None and "lbl_logits" in self.output:
            result['label'] = {'lbl_loss': self.lbl_loss_fn(self.output['lbl_logits'], label)}
            
        # 2. Annotation (Pixel-wise) Loss
        if annotation is not None and "ann_logits" in self.output:
            seg_logits = self.output['ann_logits']
            B = seg_logits.size(0)  # (B, N, C)
            
            total_tile = 0.0
            total_dice = 0.0
            for b in range(B):
                pred_nc = seg_logits[b]         # (N, C_ann)
                ann_nc = annotation[b]          # (N, C_ann)
    
                # Per-tile label loss
                total_tile = total_tile + self.ann_loss_fn(pred_nc, ann_nc)
                # Dice loss: transpose to (N, C) for per-class across N tiles
                total_dice = total_dice + self.dice_fn(pred_nc.t(), ann_nc.t())
    
            tile_loss = total_tile / B
            dice_loss = total_dice / B
            ann_loss = self.ann_weight * tile_loss + self.dice_weight * dice_loss
    
            result.update(annotation={'ann_loss': ann_loss, 'tile_loss': tile_loss, 'dice_loss': dice_loss})
            
        return result
        
    @torch.no_grad()
    def predict(self, attention: torch.Tensor = None, lbl_logits: torch.Tensor = None, ann_logits: torch.Tensor = None, **kwargs) -> dict:
        if lbl_logits is None: lbl_logits = self.output.get('lbl_logits')
        if attention is None: attention = self.output.get('attention')
        if ann_logits is None: ann_logits = self.output.get('ann_logits')
        
        self.result = {}
        
        if lbl_logits is not None:
            pred_lbl = apply_pred(self.lbl_mode, lbl_logits)
            self.result['pred_lbl'] = to_cpu(pred_lbl)
            
        if attention is not None:
            self.result['pred_atn'] = to_cpu(attention)
            
        if ann_logits is not None:
            pred_ann = apply_pred(self.ann_mode, ann_logits)
            self.result['pred_ann'] = to_cpu(pred_ann)
            
        return self.result
        
    def metric(self, label: torch.Tensor = None, annotation: torch.Tensor = None, pred_lbl: torch.Tensor = None, pred_ann: torch.Tensor = None, threshold: float = 0.5, **kwargs) -> dict:
        if pred_lbl is None: pred_lbl = self.result.get('pred_lbl')
        if pred_ann is None: pred_ann = self.result.get('pred_ann')
        
        result = {}
        
        # 1. Annotation (Pixel-wise) Metrics
        if annotation is not None and pred_ann is not None:
            annotation = torch.as_tensor(annotation).float()
            result.update(annotation=self.ann_metric(pred_ann, annotation, threshold))
            
        # 2. Slide-level Metrics
        if label is not None and pred_lbl is not None:
            label = torch.as_tensor(label).float()
            result.update(label=self.lbl_metric(pred_lbl, label, threshold))
            
        return result