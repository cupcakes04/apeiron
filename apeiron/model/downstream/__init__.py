from .abmil import ABMIL
from .gatmil import GATMIL
from .classifier import MLPClassifier
from .contrastive import ContrastiveVLM
from .generative import GenerativeVLM
from .detr import SparseDETR
from .spconv import SparseUNet
from .unet import UNet
from .roimil import ROIMIL
from .mask2f import Mask2FormerMIL

"""
Downstream Model API Contract
=============================
All new downstream models must implement the following core methods and adhere to the 
specified input/output formats to ensure compatibility with the `Inferencer` and `Composite` models.

1. `forward(self, features: torch.Tensor, **kwargs) -> dict`
   - Inputs: 
     - `features`: (B, N, F) for tile-based models or (B, F) for slide-level models.
     - Additional modalities or metadata passed via `**kwargs`.
   - Output: A dictionary containing raw unnormalized outputs (must also be saved to `self.output`).
     Keys map to `HeadData` attributes:
     - `lbl_logits`: (B, C) slide-level class logits
     - `ann_logits`: (B, N, C) tile-level annotation logits
     - `attention`: (B, 1, N) or (B, C, N) attention scores
     - `obj_classes`: (B, Q, C+1) per-query class logits (includes 'no-object' class)
     - `obj_masks`: (B, Q, N) per-query binary mask logits over N tiles
     - `img_emb`: (B, H) image embeddings for contrastive learning
     - `vis_emb`: (B, H) pooled visual embeddings for generative text modeling

2. `loss(self, **kwargs) -> dict`
   - Inputs: Ground truth tensors (`label`, `annotation`, `text`, etc.) and optionally explicit raw outputs.
     If outputs are not provided in kwargs, they should be retrieved from `self.output`.
   - Output: A nested dictionary mapped to objective categories (defined in `Objectives`).
     Example: `{'label': {'lbl_loss': tensor(0.5)}, 'annotation': {'ann_loss': tensor(0.3)}}`

3. `predict(self, **kwargs) -> dict`
   - Inputs: Raw outputs from `forward` (can fallback to `self.output`).
   - Output: A dictionary of post-processed predictions as detached numpy arrays on CPU (saved to `self.result`).
     Keys must map to `PredData` attributes:
     - `pred_lbl`: (B, C) probabilities/predictions for slide
     - `pred_ann`: (B, N, C) probabilities/predictions for tiles
     - `pred_atn`: (B, 1, N) normalized attention weights
     - `pred_obj`: list (length B) of lists, where each inner list contains dicts representing detected objects: `{"class": int, "mask": np.ndarray shape (N,) boolean, "scores": float}`
     - `pred_txt`: list[str] (length B) of generated text strings

4. `metric(self, threshold: float = 0.5, **kwargs) -> dict`
   - Inputs: Ground truth tensors, `threshold`, and predictions from `predict` (can fallback to `self.result`).
   - Output: A nested dictionary of evaluation metrics grouped by objective categories.
     Example: `{'label': {'auc': 0.85, 'f1': 0.82}, 'annotation': {'dice': 0.75}}`
"""