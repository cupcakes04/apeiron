"""Downstream task models consuming Collector output.

Models:
    ABMIL: Attention-based MIL for slide-level classification with per-tile attention maps.
    SparseUNet: UNet operating on sparse tile features with coordinates.
    SparseDETR: DETR-style segmentation on sparse tile features with Hungarian matching.
    MLPClassifier: Simple MLP classifier for standalone tile features.
"""

MODALITIES = ['label', 'annotation', 'objects', 'text']

from .MIL import ABMIL, GATMIL
from .SEG import SparseUNet
from .OBJ import SparseDETR
from .CLS import MLPClassifier, unsqueeze_outer_batch, squeeze_outer_batch
from .VLM import GenerativeVLM, ContrastiveVLM, VLM_LOSS_TYPES
from .loss import *
