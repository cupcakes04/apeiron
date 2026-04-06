from torch.utils.data import Dataset
import torch
import openslide
import numpy as np

class SlideTiles(Dataset):
    """PyTorch Dataset for loading tiles from whole slide images.
    
    Uses OpenSlide for efficient tile extraction with lazy slide opening
    to support multi-worker data loading. Each worker opens its own slide
    instance to avoid threading issues.
    
    Args:
        slide_path (str or Path): Path to whole slide image file
        tile_coords (np.ndarray): (N, 2) array of tile top-left coordinates
        encoder (int): Tile size in pixels. Default 224
        transform (callable, optional): Image preprocessing function
    
    Attributes:
        slide_path: Path to slide file
        tile_coords: Tile coordinate array
        encoder: Tile size
        transform: Preprocessing function
        _slide: Lazily opened OpenSlide object (per worker)
    """
    def __init__(self, slide_path, tile_coords, encoder=224, transform=None):
        self.slide_path = slide_path
        self.tile_coords = tile_coords
        self.encoder = encoder
        self.transform = transform
        self._slide = None  # opened lazily per worker

    def _get_slide(self):
        """Lazily open slide for current worker process.
        
        Returns:
            OpenSlide: Opened slide object
        """
        if self._slide is None:
            self._slide = openslide.open_slide(self.slide_path)
        return self._slide

    def __len__(self):
        """Get number of tiles in dataset.
        
        Returns:
            int: Number of tiles
        """
        return len(self.tile_coords)

    def __getitem__(self, idx):
        """Load and preprocess a single tile.
        
        Args:
            idx (int): Tile index
        
        Returns:
            tuple: (image, coordinates) where:
                - image: Preprocessed tile tensor if transform provided, else PIL Image
                - coordinates: Tile coordinate array
        """
        slide = self._get_slide()
        x, y = self.tile_coords[idx][:2]
        x, y = int(x), int(y)
        
        image = slide.read_region(
            (x, y),
            level=0,
            size=(self.encoder, self.encoder)
        ).convert("RGB")
        
        if self.transform:
            image = self.transform(image).squeeze(dim=0)
        else:
            # If no transform, manually convert to tensor [C, H, W]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        return image, torch.tensor([x, y], dtype=torch.int32)