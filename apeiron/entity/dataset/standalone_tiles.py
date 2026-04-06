import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class StandaloneTiles(Dataset):
    """PyTorch Dataset for loading standalone tile images without windowing.
    
    Processes individual tile images directly without extracting sub-windows.
    Each tile is treated as a single sample. Used when tiles are already at
    the desired resolution and size for feature extraction.
    
    Args:
        tile_paths (list or np.ndarray): List of file paths to tile images
        transform (callable, optional): Image preprocessing function (e.g., model transform)
    
    Attributes:
        tile_paths (list): List of tile image paths
        transform: Preprocessing transformation function
    """
    def __init__(self, tile_paths, transform=None):
        # Convert to list to ensure easy indexing
        self.tile_paths = list(tile_paths)
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        """Load and preprocess a single tile image.
        
        Args:
            idx (int): Index of tile to load
        
        Returns:
            tuple: (image, dummy_coords, has_coords) where:
                - image: Preprocessed tile tensor (C, H, W)
                - dummy_coords: Placeholder tensor [-1, -1] (no spatial coords)
                - has_coords: False (standalone tiles have no coordinates)
        """
        # 1. Get the path
        img_path = self.tile_paths[idx]
        
        # 2. Load the image using PIL (compatible with torch transforms)
        image = Image.open(img_path).convert("RGB")
    
        # 3. Apply transformations to get (C, H, W) tensor
        if self.transform:
            image = self.transform(image).squeeze(dim=0)
        else:
            # If no transform, manually convert to tensor [C, H, W]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # 4. Return with dummy coordinates (standalone tiles have no spatial context)
        return image, torch.tensor([0, 0], dtype=torch.int32)