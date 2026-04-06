import torch
from .backbone import Backbone
from apeiron.entity.dataset import SlideTiles, StandaloneTiles, WindowTiles
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Literal
from apeiron.entity.process.helper.tokens import aggregate_patch_tokens

class Extractor:
    """Handles feature extraction from whole slide images using foundational models.
    
    Processes slide tile_array through vision transformer models to generate embeddings.
    Supports both class tokens and patch tokens for multi-scale feature extraction.
    
    Args:
        **kwargs: Additional keyword arguments passed to parent classes
    
    Attributes:
        embeddings (dict): Extracted features containing:
            - 'class_token': (N, F) global tile-level features
            - 'patch_tokens': (N, 256, F) local patch-level features
            - 'max_patch_token': (N, F) max-pooled patch features
            - 'mean_patch_token': (N, F) mean-pooled patch features
        dataset (SlideTiles): PyTorch dataset for tile loading
        device: Computation device from Backbone
        model: Neural network model from Backbone
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extract_dataset: SlideTiles | StandaloneTiles | WindowTiles
        self.device: Backbone.device
        self.model: Backbone.model
        self.reset_embeddings()
        
    def reset_embeddings(self):
        """Clear all stored embeddings to free memory."""
        self.embeddings = {"coords": [], "class_token": [], "patch_tokens": [], "max_patch_token": [], "mean_patch_token": []}
        
    def load_embeddings(self, embeddings):
        """Load generated embeddings."""
        self.embeddings = embeddings

    def extract(self, tile_array, tile_coord, ext_patch_strategy):
        """
        Wrap with this before calling:
        ```python
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
        ```
        """
        # 1. torch.float16 to match autocast
        tile_array = tile_array.to(self.device, non_blocking=True, dtype=torch.float16)
        
        # 2. Forward pass (B, 3, 224, 224)
        class_token, patch_tokens = self.model(tile_array)
        
        # 3. Collect results (moving to CPU to save VRAM)
        self.embeddings['coords'].append(tile_coord)
        self.embeddings['class_token'].append(class_token.cpu())     # (N, F)
        
        # 4. Patch Operations to aggregate patch tokens (save less to disk)
        if ext_patch_strategy == 'aggr':
            self.embeddings['max_patch_token'].append(aggregate_patch_tokens(patch_tokens.cpu(), 'max', is_torch=True))
            self.embeddings['mean_patch_token'].append(aggregate_patch_tokens(patch_tokens.cpu(), 'mean', is_torch=True))
                
        elif ext_patch_strategy in ["keep", True]:
            self.embeddings['patch_tokens'].append(patch_tokens.cpu())     # (N, 256, F)
            
        elif ext_patch_strategy in ["discard", False]:
            pass

    def extract_tiles(self, batch_size=300, num_workers=4, ext_patch_strategy=Literal["aggr", "discard", "keep"]):
        """Extract embeddings from slide (getting tile_arrays using the loaded foundational model.
        
        Processes tile_array in batches through the model to generate features. Supports different
        strategies for handling patch tokens to balance memory usage and feature granularity.
        
        Args:
            batch_size (int): Number of tile_array to process simultaneously. Default 300
            num_workers (int): Number of parallel data loading workers. Default 4
            ext_patch_strategy (str): Strategy for patch token handling:
                - 'discard': Only keep class tokens (lowest memory, N×F)
                - 'keep': Keep all patch tokens (highest memory, N×256×F)
                - 'aggr': Aggregate patches via max/mean pooling (medium memory, N×2F)
        
        Returns:
            dict: Embeddings dictionary with keys depending on ext_patch_strategy:
                - Always includes 'class_token': (N, F) array
                - 'keep' adds 'patch_tokens': (N, 256, F) array
                - 'aggr' adds 'max_patch_token' and 'mean_patch_token': (N, F) arrays
        """
        
        loader = DataLoader(self.extract_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                for tile_array, tile_coord in tqdm(loader):
                    self.extract(tile_array, tile_coord, ext_patch_strategy)

        self.embeddings = self._concat_tonumpy_embeddings(self.embeddings)
        return self.embeddings
                        
    @staticmethod
    def _concat_tonumpy_embeddings(embeddings):
        # Final Concatenation - convert to numpy arrays with float32
        final_embeddings = {}
        for k, v in embeddings.items():
            if len(v) > 0:
                if k == 'coords':
                    final_embeddings[k] = torch.cat(v, dim=0).numpy().astype(np.int32)
                else:
                    final_embeddings[k] = torch.cat(v, dim=0).numpy().astype(np.float32)
        return final_embeddings