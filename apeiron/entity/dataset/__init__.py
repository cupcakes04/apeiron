"""PyTorch Dataset classes for loading tiles from various sources.

Provides three dataset types:
    SlideTiles: Extract tiles on-the-fly from whole slide images via OpenSlide.
    StandaloneTiles: Load pre-existing tile image files directly.
    WindowTiles: Extract sub-windows from large tile images (pseudo-slide mode).
"""

from .slide_tiles import *
from .standalone_tiles import *
from .window_tiles import *