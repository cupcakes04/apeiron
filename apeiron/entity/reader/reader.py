import json
import numpy as np
from pathlib import Path
from sklearn.utils import resample
import openslide
from .thumbnail import Thumbnailer
from tiatoolbox.wsicore.wsireader import WSIReader
from .mask import full_mask_to_tile_coords, mask_to_tile_coords, show_masked_tiles
import logging
logging.getLogger().setLevel(logging.ERROR)

class Reader(Thumbnailer):
    """Handles whole slide image and tile image reading, plus tile coordinate generation.

    Manages slide opening, metadata extraction, resolution scaling, and tile
    coordinate generation for feature extraction.  Also supports opening
    standalone tile images and computing windowed tile coordinates.
    Inherits thumbnail and stain-normalization capabilities from Thumbnailer.

    Args:
        ext_enc (int): Encoder window size in pixels at ``ext_mpp`` resolution. Default 224.
        ext_mpp (float): Target microns per pixel for feature extraction. Default 0.5.
        **kwargs: Additional keyword arguments passed to Thumbnailer.

    Attributes:
        ext_enc (int): Encoder size at extraction resolution (before scaling).
        ext_mpp (float): Target microns per pixel for extraction.
        encoder (int): Actual encoder size in pixels at the slide's base resolution,
            computed as ``ceil(ext_enc * ext_mpp / base_mpp)``.
        slide_path (Path): Path to the currently opened slide file.
        slide_name (str): Slide filename without extension.
        slide_obj (WSIReader): TIAToolbox WSI reader instance for the open slide.
        tile_coords (np.ndarray): (N, 2) or (N, 3) tile coordinates for extraction.
        tile_paths (list): List of tile image paths (tile mode only).
    """
    def __init__(self, ext_enc=224, ext_mpp=0.5, **kwargs):
        super().__init__(**kwargs)
        self.update_res_unit(ext_enc, ext_mpp)
        self.encoder = None
        self.slide_path = None
        self.slide_name = None
        self.slide_obj: WSIReader
        self.tile_coords = None

    def update_res_unit(self, ext_enc=None, ext_mpp=None):
        """Update extraction resolution parameters.

        Args:
            ext_enc (int, optional): New encoder window size in pixels.
            ext_mpp (float, optional): New target microns per pixel.
        """
        if ext_enc:
            self.ext_enc = ext_enc
        if ext_mpp:
            self.ext_mpp = ext_mpp
        
    def setup_slide(self, slide_path, base_mpp=None):
        """Open a whole slide image and initialize slide-specific attributes.
        
        Closes any previously opened slide to prevent memory leaks. Calculates
        encoder size based on resolution scaling.
        
        Args:
            slide_path (str or Path): Path to slide file
            base_mpp (float, optional): Base microns per pixel. Auto-detected if None
        """
        
        # Close any previously opened slide
        try:
            if slide_path == self.slide_path:
                return
            elif self.slide_path != None:
                self.slide_obj.openslide_wsi.close()
        except:
            pass
        
        self.slide_path = Path(slide_path)
        self.slide_name = self.slide_path.stem
        
        self.slide_obj = WSIReader.open(self.slide_path)
        self.width, self.height = self.slide_obj.slide_dimensions(0, 'level')
        self.dimensions_hw = [self.height, self.width]
        
        if base_mpp:
            self.base_mpp = base_mpp
        else:
            self.info = self.slide_obj.info.as_dict()
            self.base_mpp = self.info['mpp'][0]

        self.encoder = self._calc_encoder_val()
        self.tile_coords = None

    def setup_tiles(self, tile_paths, base_mpp=None):
        """Initialize tile-mode attributes for standalone or windowed tile analysis.

        Args:
            tile_paths (list[str | Path]): List of file paths to tile images.
            base_mpp (float, optional): Base microns per pixel for the tiles.
        """
        self.tile_paths = tile_paths
        
        self.base_mpp = base_mpp
        self.encoder = self._calc_encoder_val()
        self.tile_coords = None
        
    def _calc_encoder_val(self) -> int:
        """Compute the encoder tile size in pixels at the slide's base resolution.

        Returns:
            int: Scaled encoder size = ceil(ext_enc * ext_mpp / base_mpp).
        """
        return int(np.ceil(self.ext_enc * (self.ext_mpp / self.base_mpp)))

    
    def read_window(self, coord, resolution=0, units='level', encoder=None, normalise=False):
        """Read a single tile from the whole slide image.
        
        Args:
            coord (array-like): Top-left coordinate [x, y] or [W, H]
            resolution (int): Pyramid level to read from. Default 0 (highest resolution)
            units (str): Coordinate units ('level', 'mpp', 'power'). Default 'level'
            encoder (int, optional): Tile size override. Uses self.encoder if None
        
        Returns:
            np.ndarray: RGB image tile of shape (encoder, encoder, 3)
        """
        x, y = coord[:2]
        _encoder = encoder if encoder else self.encoder
        
        image = self.slide_obj.read_bounds(
            (int(x), int(y), int(x)+_encoder, int(y)+_encoder), 
            resolution=resolution, 
            units=units
        )
        # Can be set to true to run normalisation
        if normalise:
            image = self.normalise_image(image)
        return image
    
    def create_tile_coords(self, method, ann_mask=False, tile_threshold=0.25, stride=None):
        """Generate tile coordinates for the currently opened slide.

        Supports three strategies:
        - **Annotation mask**: Use an annotation file to restrict tiles to annotated regions.
        - **Full extraction**: Generate a regular grid covering the entire slide.
        - **Tissue mask**: Use the binary tissue mask to skip background tiles.

        Args:
            method (str): Masking method for tissue detection (e.g. ``'morphological'``,
                ``'otsu'``, ``'full'``). ``'full'`` skips masking entirely.
            ann_mask (str or bool): Path to an annotation file to use as a mask,
                or ``False`` to disable annotation-based masking.
            tile_threshold (float): Minimum tissue ratio (0–1) for a tile to be
                accepted when using tissue masking. Default 0.25.
            stride (int, float, or None): Tile overlap specification.
                See :func:`resolve_stride` for details.

        Returns:
            np.ndarray: (N, 2) array of tile top-left coordinates [x, y].
        """
        
        # Case A: Annotation as mask
        if ann_mask and Path(ann_mask).is_file():
            tile_coords = mask_to_tile_coords(
                slide_w=self.width, slide_h=self.height, 
                encoder=self.encoder, mask=self.masked_thumbnail,
                tile_threshold=tile_threshold, stride=stride
            )
            self.prepare_annotations(tile_coords, self.encoder, ann_path=ann_mask, active_coords=True)
            tile_coords = tile_coords[self.annotation]

        # Case B: All-Tile Extraction
        elif method == 'full':
            tile_coords = full_mask_to_tile_coords(
                slide_w=self.width, slide_h=self.height, 
                encoder=self.encoder, stride=stride
            )

        # Case C: Masked Extraction
        else:
            tile_coords = mask_to_tile_coords(
                slide_w=self.width, slide_h=self.height, 
                encoder=self.encoder, mask=self.masked_thumbnail,
                tile_threshold=tile_threshold, stride=stride
            )
        
        self.tile_coords = tile_coords.astype(np.int32)
        return self.tile_coords

        
    def create_windowed_tile_coords(self, stride=None):
        """Precompute all window coordinates for a set of tile images.
        
        Generates a mapping of all windows to be extracted from each tile,
        including the tile index to enable efficient batch processing across
        multiple tiles.
        
        Args:
            tile_paths (list): List of paths to tile images
            encoder (int): Window size in pixels
            stride (float): Overlap ratio in [0, 1]. 0 = no overlap, 0.5 = 50% overlap
        
        Returns:
            np.ndarray: (N, 3) array where each row is [tile_idx, x, y]
                - tile_idx: Index into tile_paths
                - x, y: Top-left coordinates of window in the tile
        """
        windowed_tile_coords = []
        for idx, path in enumerate(self.tile_paths):
            # Get tile dimensions
            with openslide.open_slide(path) as slide:
                w, h = slide.dimensions
                
            # Generate all window coordinates for this tile
            coords = full_mask_to_tile_coords(w, h, self.encoder, stride)
            
            # Add tile index to each coordinate for mapping back to source
            for x, y in coords:
                windowed_tile_coords.append((idx, int(x), int(y)))
                
        # Shape: [Total_Windows, 3]
        self.tile_coords = np.array(windowed_tile_coords).astype(np.int32)
        return self.tile_coords
        
        
        