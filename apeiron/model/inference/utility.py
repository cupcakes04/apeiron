import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal, List
from ..downstream import *
from dataclasses import dataclass, field


MODALITIES = ['label', 'annotation' , 'objects', 'text']
LOSS_MODALITIES = ['lbl_loss', 'ann_loss', 'obj_loss', 'gen_loss', 'con_loss']
INF_MODELS = ['abmil', 'gatmil', 'roimil', 'mask2f', 'classifier', 'unet', 'spconv', 'detr', 'generative', 'contrastive']
            
@dataclass
class HeadData:
    attention: np.ndarray | None = None      # (B, 1, N) or (B, C, N) attention scores
    lbl_logits: np.ndarray | None = None     # (B, C) slide-level class logits
    ann_logits: np.ndarray | None = None     # (B, N, C) tile-level annotation logits
    obj_classes: np.ndarray | None = None    # (B, Q, C+1) per-query class logits (includes 'no-object' class)
    obj_masks: np.ndarray | None = None      # (B, Q, N) per-query binary mask logits
    vis_emb: np.ndarray | None = None        # (B, H) pooled visual embeddings for text generation
    img_emb: np.ndarray | None = None        # (B, H) image embeddings for contrastive learning

@dataclass
class PredData:
    pred_atn: np.ndarray | None = None       # (B, 1, N) normalized attention weights
    pred_lbl: np.ndarray | None = None       # (B, C) probabilities/predictions for slide
    pred_ann: np.ndarray | None = None       # (B, N, C) probabilities/predictions for tiles
    pred_obj: list | None = None             # list (length B) of lists containing dicts: {"ids": list[int], "labels": np.ndarray(C,), "scores": float}
    pred_txt: list | None = None             # list[str] (length B) of generated text strings
    
    # Post-processed predictions
        # label (C)
        # annotation (N, C)
        # attention (N, C)
        # objects (Q, N, C)
        # scores(Q)
    post_processed: bool = False
    pred_crd: np.ndarray | None = None
    pred_scr: np.ndarray | None = None
    pred_data_type: str | None = None   # 'single' or 'group'

@dataclass
class Objectives:
    label: dict = field(default_factory=dict)
    annotation: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)

    def get_dict(self, **kwargs):
        return {
            'label': self.label,
            'annotation': self.annotation,
            'objects': self.objects,
            'text': self.text,
            **kwargs
        }

@dataclass
class ModelData:

    model_names: dict | None = None
    composite: dict = field(default_factory=lambda: {'final_loss': 0})

    head: HeadData = field(default_factory=HeadData)
    pred: PredData = field(default_factory=PredData)
    loss: Objectives = field(default_factory=Objectives)
    metric: Objectives = field(default_factory=Objectives)

    def assign(self, mode: Literal['head', 'pred', 'loss', 'metric'], **kwargs):
        mode_map = {
            'head': self.head,
            'pred': self.pred,
            'loss': self.loss,
            'metric': self.metric
        }
        obj = mode_map.get(mode)

        for key, value in kwargs.items():
            if not hasattr(obj, key):
                continue

            if isinstance(value, dict):
                target_obj_dict = getattr(obj, key)
                if len(target_obj_dict) == 0:
                    setattr(obj, key, value)
                else:
                    target_obj_dict.update(value)
            else:
                setattr(obj, key, value)

    def composite_loss(self, final_loss, **importance):
        self.composite = {
            'final_loss': final_loss,
            **importance,
        }


def choose_inferencer(
    in_features, mode=None, inf_model='abmil',
    lbl_n_classes=0, lbl_loss_type: str = 'hard_ce', lbl_cls_weights: dict = None, 
    ann_n_classes=0, ann_loss_type: str = 'bce', ann_cls_weights: dict = None):

    if inf_model == 'abmil':
        inferencer = ABMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            dropout = 0.25,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
        )

    elif inf_model == 'gatmil':
        inferencer = GATMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            k_neighbors = 32,
            num_heads = 4,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
        )

    elif inf_model == 'roimil':
        inferencer = ROIMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            dropout = 0.25,
            attn_dim = 128,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
        )
        
    elif inf_model == 'mask2f':
        inferencer = Mask2FormerMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            dropout = 0.25,
            attn_dim = 128,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            num_heads = 4,
            num_layers = 2,
        )

    elif inf_model == 'classifier':
        inferencer = MLPClassifier(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            hidden_dim = 256,
            n_layers = 2,
            dropout = 0.25,
            mode = mode,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
        )

    elif inf_model == 'unet':
        inferencer = UNet(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            base_channels = 256,
            depth = 3,
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights,
        )

    elif inf_model == 'spconv':
        inferencer = SparseUNet(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            base_channels = 256,
            depth = 3,
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights,
        )

    elif inf_model == 'detr':
        inferencer = SparseDETR(
            in_features = in_features,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            n_heads = 8,
            n_encoder_layers = 3,
            n_decoder_layers = 3,
            n_queries = 20,
            dropout = 0.1,
            ann_cls_weights = ann_cls_weights,
            ann_loss_type = ann_loss_type,
        )

    elif inf_model == 'generative':
        inferencer = GenerativeVLM(
            in_features = in_features,
            lm_model_name = "distilgpt2",
            num_visual_tokens=32,
            use_lora=True,
            mode = mode,
        )

    elif inf_model == 'contrastive':
        inferencer = ContrastiveVLM(
            in_features=in_features,
            text_model_name = "emilyalsentzer/Bio_ClinicalBERT",
            num_visual_tokens=32,
            projection_dim=512,
            mode = mode,
        )

    return inferencer
