import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import openslide

class WindowTiles(Dataset):
    """PyTorch Dataset for extracting windows from tile images.
    
    Treats tile images like mini whole-slide images by extracting overlapping
    or non-overlapping windows at specified encoder size. Useful for processing
    large tiles that need to be divided into smaller patches for feature extraction.
    
    Args:
        tile_paths (list): List of paths to tile images
        windowed_tile_coords (np.ndarray): (N, 3) array of [tile_idx, x, y] coordinates
        encoder (int): Window size in pixels. Default 224
        transform (callable, optional): Image preprocessing function
    
    Attributes:
        tile_paths (list): Tile image paths
        windowed_tile_coords (np.ndarray): Window coordinate mapping
        encoder (int): Window extraction size
        transform: Preprocessing function
        _slide: Cached OpenSlide object for current tile (worker-specific)
        _cur_idx: Index of currently opened tile (worker-specific)
    """
    def __init__(self, tile_paths, windowed_tile_coords, encoder=224, transform=None):
        self.tile_paths = tile_paths
        self.windowed_tile_coords = windowed_tile_coords
        self.encoder = encoder
        self.transform = transform
        
        # Worker-specific cache for lazy loading
        self._slide = None
        self._cur_idx = None

    def __len__(self):
        return len(self.windowed_tile_coords)

    def __getitem__(self, idx):
        """Load and preprocess a window from a tile image.
        
        Lazily opens tiles as needed and caches the current tile to avoid
        repeated file I/O when processing multiple windows from the same tile.
        
        Args:
            idx (int): Window index
        
        Returns:
            tuple: (image, coordinates, has_coords) where:
                - image: Preprocessed window tensor (C, H, W)
                - coordinates: Tensor [x, y, tile_idx] for window location
                - has_coords: True (windowed tiles have spatial coordinates)
        """
        # Extract metadata from coordinate mapping [tile_idx, x, y]
        path_idx, x, y = self.windowed_tile_coords[idx]
        x, y = int(x), int(y)

        # Lazy open: only open new tile if different from current
        if self._slide is None or self._cur_idx != path_idx:
            if self._slide: 
                self._slide.close()
            self._slide = openslide.open_slide(self.tile_paths[path_idx])
            self._cur_idx = path_idx

        # Extract window at specified coordinates
        image = self._slide.read_region(
            (x, y),
            level=0,
            size=(self.encoder, self.encoder)
        ).convert("RGB")
        
        # Apply preprocessing transforms
        if self.transform:
            image = self.transform(image).squeeze(dim=0)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        return image, torch.tensor([x, y, path_idx], dtype=torch.int32)