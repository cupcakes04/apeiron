import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, List
from apeiron.utils import convert_to_list
from dataclasses import dataclass, field

from ..downstream import *
from .predictor import *

def choose_inferencer(in_features, mode=None, lbl_n_classes=0, ann_n_classes=0, inf_model='abmil'):

    if inf_model == 'abmil':
        inferencer = ABMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            dropout = 0.25,
        )

    if inf_model == 'gatmil':
        inferencer = GATMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            k_neighbors = 32,
            num_heads = 4,
        )

    elif inf_model == 'mlp':
        inferencer = MLPClassifier(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            hidden_dim = 256,
            n_layers = 2,
            dropout = 0.25,
            mode = mode,
        )

    elif inf_model == 'unet':
        inferencer = SparseUNet(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            base_channels = 256,
            depth = 3,
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
        )

    elif inf_model == 'genvl':
        inferencer = GenerativeVLM(
            in_features = in_features,
            lm_model_name = "distilgpt2",
            num_visual_tokens=32,
            use_lora=True,
            mode = mode,
        )

    elif inf_model == 'convl':
        inferencer = ContrastiveVLM(
            in_features=in_features,
            text_model_name = "emilyalsentzer/Bio_ClinicalBERT",
            num_visual_tokens=32,
            projection_dim=512,
            mode = mode,
        )

    return inferencer

@dataclass
class HeadData:
    attention: np.ndarray | None = None
    lbl_logits: np.ndarray | None = None
    ann_logits: np.ndarray | None = None
    obj_classes: np.ndarray | None = None
    obj_masks: np.ndarray | None = None
    gen_loss: np.ndarray | None = None
    con_loss: np.ndarray | None = None
    gen_fn: np.ndarray | None = None

@dataclass
class PredData:
    pred_atn: np.ndarray | None = None
    pred_lbl: np.ndarray | None = None
    pred_ann: np.ndarray | None = None
    pred_obj: list | None = None
    pred_txt: str | None = None
    
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
    modalities: set | None = None
    composite: dict = field(default_factory=lambda: {'final_loss': 0})

    head: HeadData = field(default_factory=HeadData)
    pred: PredData = field(default_factory=PredData)
    loss: Objectives = field(default_factory=Objectives)
    metric: Objectives = field(default_factory=Objectives)

    def assign(self, mode: Literal['head', 'pred', 'loss', 'metric'], **kwargs):
        if mode == 'head'   : obj = self.head
        if mode == 'pred'   : obj = self.pred
        if mode == 'loss'   : obj = self.loss
        if mode == 'metric' : obj = self.metric
        for key, value in kwargs.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def composite_loss(self, final_loss, **importance):
        self.composite = {
            'final_loss': final_loss,
            **importance,
        }


"""Composite downstream models that merge two task heads behind a shared encoder.

Valid combinations:
    - **CLS+MIL** (``CLSMIL``): Per-tile MLP classification + slide-level ABMIL.
      Both heads share a feature encoder. The MLP classifies individual tiles
      while ABMIL aggregates for slide-level labels + attention.

    - **MIL+SEG** (``MILSEG``): Slide-level ABMIL + annotation-level segmentation
      (UNet or DETR). Both heads share a feature encoder. ABMIL produces slide
      labels + attention, while the seg head produces per-tile annotation logits.

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
        lbl_n_classes=0, ann_n_classes=0):
        super().__init__()

        # 1. Use our 1536-D Residual Encoder from before
        self.encoder = SharedEncoder(in_features=in_features)
        
        # 2. Register heads as a ModuleDict so PyTorch can see their weights
        heads_dict = {}
        for inf_model in convert_to_list(inf_models):
            inferencer = choose_inferencer(in_features, mode, lbl_n_classes, ann_n_classes, inf_model)
            heads_dict[inf_model] = inferencer
        self.heads = nn.ModuleDict(heads_dict)
        
        # 3. Extract Model metadata
        self.modalities = set()
        self.embedding_fns = {}
        for name, head_module in self.heads.items():
            model_info = head_module.get_model_info()
            self.modalities.update(model_info['modality'])

            # Any Model utilities is updated here
            self.embedding_fns = model_info.get('embedding_fns', self.embedding_fns)


    def forward(self, features, **kwargs) -> ModelData:
        """
        Args:
            head_outputs: Dict containing sub-model outputs:
                {
                   'model_A': {'seg_logits': (B, C_seg, N), 'attention': (B, C_lbl, N), instances: (B, N, D)},
                   'model_B': {'seg_logits': (B, C_seg, N), 'attention': (B, C_lbl, N), instances: (B, N, D)}
                }
        """
    ### (A) Run Model Heads
        h = self.encoder(features)  # (1536 -> 1536)
        head_outputs = {name: head_module(h, **kwargs) for name, head_module in self.heads.items()}
            
    ### (B) Unify Heads
        all_mil_a, all_mil_i = [], []
        all_cls, all_seg = [], []
        all_det_c, all_det_m = [], []
        txt_loss, txt_fn = {}, {}

        mdata = ModelData(model_names=list(head_outputs.keys()), modalities=self.modalities)
        for out in head_outputs.values():
            mdata.assign(mode='head', **out)
        return mdata


class MultiModalLoss(nn.Module):
    """
    A dynamic multi-head model using a shared residual encoder.
    
    Args:
        heads_dict (dict): A dictionary where keys are task names (str) 
                           and values are the head modules (nn.Module).
    """
    def __init__(self, 
        lbl_loss_type=None, ann_loss_type=None, 
        lbl_cls_weights: dict = None, ann_cls_weights: dict = None):
        super().__init__()

        # The 3 label modalities
        self.lbl_losser = LabelLoss(lbl_loss_type=lbl_loss_type, lbl_cls_weights=lbl_cls_weights)
        self.ann_losser = AnnotationLoss(ann_loss_type=ann_loss_type, ann_cls_weights=ann_cls_weights)
        self.obj_losser = ObjectsLoss(ann_cls_weights=ann_cls_weights)
        self.txt_losser = TextLoss(loss_types=VLM_LOSS_TYPES)
        
        # Initialize AWL with the specific keys we expect
        self.awl = AutomaticWeightedLoss(task_keys=MODALITIES)

    def forward(self, mdata: ModelData, **kwargs) -> ModelData:
        task_losses = {}

        # --- 1. Collect Raw Losses ---
        if mdata.head.lbl_logits is not None and kwargs.get('label') is not None:
            l_out = self.lbl_losser(mdata.head.lbl_logits, **kwargs)
            task_losses['label'] = l_out['lbl_loss']
            mdata.assign(mode='loss', label=l_out)

        if mdata.head.ann_logits is not None and kwargs.get('annotation') is not None:
            a_out = self.ann_losser(mdata.head.ann_logits, **kwargs)
            task_losses['annotation'] = a_out['ann_loss']
            mdata.assign(mode='loss', annotation=a_out)

        if all([k is not None for k in [mdata.head.obj_classes, mdata.head.obj_masks]]) \
            and any([kwargs.get(k) is not None for k in ['objects', 'annotation']]):
            d_out = self.obj_losser(mdata.head.obj_classes, mdata.head.obj_masks, **kwargs)
            task_losses['objects'] = d_out['obj_loss']
            mdata.assign(mode='loss', objects=d_out)

        if any([k is not None for k in [mdata.head.gen_loss, mdata.head.con_loss]]) \
            and kwargs.get('text') is not None:
            t_out = self.txt_losser(mdata.head.gen_loss, mdata.head.con_loss, **kwargs)
            task_losses['text'] = t_out['txt_loss']
            mdata.assign(mode='loss', text=t_out)

        # --- 2. Normalize and Weight ---
        # We pass the dict to AWL which handles the missing keys dynamically
        if task_losses:
            mdata.composite_loss(self.awl(task_losses), **self.awl.get_importance())
        return mdata

class MultiModalPredict:
    """
    A dynamic multi-head model using a shared residual encoder.
    Args:
        heads_dict (dict): A dictionary where keys are task names (str) 
                           and values are the head modules (nn.Module).
    """
    def __init__(self, lbl_loss_type=None, ann_loss_type=None) -> ModelData:
        super().__init__()
        self.lbl_predictor = LabelPredictor(lbl_loss_type=lbl_loss_type)
        self.ann_predictor = AnnotationPredictor(ann_loss_type=ann_loss_type)
        self.obj_predictor = ObjectsPredictor(ann_loss_type=ann_loss_type)
        self.txt_predictor = TextPredictor()

    def __call__(self, mdata: ModelData, target: dict = None, threshold=0.5):

        if mdata.head.lbl_logits is not None:
            l_out = self.lbl_predictor(mdata.head.lbl_logits, mdata.head.attention)
            mdata.assign(mode='pred', **l_out)
            if target and 'label' in target:
                mdata.assign(mode='metric', label=self.lbl_predictor.metrics(l_out, target, threshold))

        if mdata.head.ann_logits is not None:
            a_out = self.ann_predictor(mdata.head.ann_logits)
            mdata.assign(mode='pred', **a_out)
            if target and 'annotation' in target:
                mdata.assign(mode='metric', annotation=self.ann_predictor.metrics(a_out, target, threshold))
        
        if all([k is not None for k in [mdata.head.obj_classes, mdata.head.obj_masks]]):
            o_out = self.obj_predictor(mdata.head.obj_classes, mdata.head.obj_masks)
            mdata.assign(mode='pred', **o_out)
            if target and any([k in target for k in ['objects', 'annotation']]):
                mdata.assign(mode='metric', objects=self.obj_predictor.metrics(o_out, target, threshold))

        if mdata.head.gen_fn is not None:
            t_out = self.txt_predictor(mdata.head.gen_fn)
            mdata.assign(mode='pred', **t_out)
            if target and 'text' in target:
                mdata.assign(mode='metric', text=self.txt_predictor.metrics(t_out, target, threshold))

        return mdata
