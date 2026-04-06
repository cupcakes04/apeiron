"""APEIRON - A computational pathology framework for whole slide image analysis.

Provides end-to-end tools for:
- Reading and tiling whole slide images (WSI) and standalone tile images
- Extracting features using foundational vision transformer models
- Managing slide/tile registries and artifact storage
- Visualizing features via dimensionality reduction, clustering, and similarity scoring
- Annotating and labeling tiles with shape or pixel-level ground truth
- Collecting features for downstream training tasks

Top-level API:
    Analyzer: Low-level worker for single-slide or single-tile analysis.
    Operator: High-level interface that connects the Registry, Manager, and Analyzer
              for batch data generation and serving across projects.
    Backbone: Manages foundational model loading, caching, and selection.
"""

from .entity import *
from .utils import *
from .model import *
from .manage import *

from .analyze import Analyzer
from .operate import Operator
