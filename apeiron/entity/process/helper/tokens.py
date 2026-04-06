from typing import Literal
import math
import torch
import pandas as pd
import numpy as np

def aggregate_patch_tokens(patch_tokens, aggregate: Literal["max", "mean"], is_torch=True):
    """Aggregate patch tokens into tile-level features using pooling.
    
    Reduces patch-level tokens (e.g., 256 patches per tile) to a single
    feature vector per tile using max or mean pooling. Useful for creating
    tile-level representations from patch tokens.
    
    Args:
        patch_tokens (torch.Tensor or np.ndarray): (N, num_patches, F) patch features
            where N is number of tiles, num_patches is typically 256, F is feature dim
        aggregate (str): Pooling method - 'max' or 'mean'
        is_torch (bool): True if input is torch.Tensor, False if numpy. Default True
    
    Returns:
        torch.Tensor or np.ndarray: (N, F) aggregated tile-level features
    """
    if is_torch:
        if aggregate == 'max':
            return patch_tokens.max(dim=1).values
        elif aggregate == 'mean':
            return patch_tokens.mean(dim=1)
    else:
        if aggregate == 'max':
            return np.max(patch_tokens, axis=1)
        elif aggregate == 'mean':
            return np.mean(patch_tokens, axis=1)
        

def flatten_patch_tokens(encoder, patch_tokens, tile_coords):
    """Flatten patch tokens from tiles into individual patch features with coordinates.
    
    Converts tile-based patch tokens into a flat array of patch-level features,
    computing the spatial coordinates for each patch. Useful for fine-grained
    segmentation and patch-level analysis.
    
    Args:
        encoder (int): Tile size in pixels (e.g., 224)
        patch_tokens (np.ndarray): (N, num_patches, F) patch features from N tiles
        tile_coords (np.ndarray): (N, 2) top-left coordinates of tiles
    
    Returns:
        tuple: (patch_dim, flattened_features, flattened_coords)
            - patch_dim (int): Number of patches per dimension (e.g., 16 for 16x16 grid)
            - flattened_features (np.ndarray): (N*num_patches, F) all patch features
            - flattened_coords (np.ndarray): (N*num_patches, 2) spatial coordinates or (N*num_patches, 3) for id aware
    """
    N, num_tokens, F = patch_tokens.shape   # (N, 256, F)
    num_coord_cols = tile_coords.shape[1]
    patch_dim = int(np.sqrt(num_tokens))    # e.g., 16 for 256 patches
    tile_size = encoder / patch_dim         # Size of each patch in pixels

    # 1. Create a 1D array of offsets for patch positions within a tile
    offsets = np.arange(patch_dim) * tile_size
    
    # 2. Create a 2D grid of (x, y) offsets for all patches in one tile
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    grid_offsets = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)  # Shape: (num_tokens, 2)

    # 2. Compute absolute spatial coordinates (x, y)
    # Shape: (N, 1, 2) + (num_tokens, 2) -> (N, num_tokens, 2)
    spatial_coords = tile_coords[:, np.newaxis, :2] + grid_offsets

    # 3. Handle extra columns (like IDs) if they exist
    if num_coord_cols > 2:
        # Extract extra columns and repeat them for every patch in the tile
        # Shape: (N, 1, extra) -> (N, num_tokens, extra)
        extra_info = np.repeat(tile_coords[:, np.newaxis, 2:], num_tokens, axis=1)
        all_coords = np.concatenate([spatial_coords, extra_info], axis=-1)
    else:
        all_coords = spatial_coords

    # 4. Flatten to patch-level arrays
    return (
        patch_dim,
        patch_tokens.reshape(-1, F),
        all_coords.reshape(-1, num_coord_cols)
    )


def group_class_tokens(features, coords, tile_size, grid_size=None, min_coverage_thr=0.25):
    """Group individual tiles into larger grid blocks for multi-scale analysis.
    
    Organizes tiles into grids (e.g., 2x2 or 3x3 tile blocks) and assigns group IDs.
    Missing tiles within a grid are filled with zero features. Grids with insufficient
    coverage are discarded. This enables coarser-grained feature analysis.
    
    If coords already has 3 columns (x, y, group_id), returns inputs unchanged.
    
    Args:
        features (np.ndarray): (N, F) tile-level features
        coords (np.ndarray): (N, 2) tile top-left coordinates [x, y]
        tile_size (int): Size of individual tiles in pixels (e.g., encoder size)
        grid_size (int): Number of tiles per dimension in each grid (e.g., 2 for 2x2)
        min_coverage_thr (float): Minimum fraction of tiles required in a grid to keep it.
            E.g., 0.25 means at least 25% of expected tiles must be present. Default 0.25
    
    Returns:
        tuple: (features_out, coords_out)
            - features_out (np.ndarray): (M, F) features with zero-padding for missing tiles
            - coords_out (np.ndarray): (M, 3) coordinates with [x, y, group_id]
                where M = num_valid_grids x grid_size²
    
    Example:
        For grid_size=2, each grid contains 4 tiles (2x2). If a grid has only 3 tiles,
        it's kept if min_coverage_thr ≤ 0.75, with the 4th tile zero-filled.
    """
    if coords.shape[1] > 2:
        return features, coords

    num_tiles, feat_dim = features.shape
    
    # 1. Map tile coordinates to integer indices
    tx = (coords[:, 0] // tile_size).astype(int)
    ty = (coords[:, 1] // tile_size).astype(int)
    
    # 2. Map tiles to their respective grids
    gx = tx // grid_size
    gy = ty // grid_size
    
    # Create a unique grid key and identify group boundaries
    # Using a 2D-to-1D mapping is faster than string concatenation
    grid_keys = np.stack([gx, gy], axis=1)
    unique_grids, inverse_indices = np.unique(grid_keys, axis=0, return_inverse=True)
    
    # 3. Fast lookup: Map (tx, ty) -> feature index
    # We use a dict of dicts or a coordinate-to-index map
    tile_to_feat_idx = {(x, y): i for i, (x, y) in enumerate(zip(tx, ty))}
    
    num_expected = grid_size * grid_size
    feats_out = []
    coords_out = []
    group_counter = 0

    # 4. Process each grid
    for g_idx, (cur_gx, cur_gy) in enumerate(unique_grids):
        # Determine the top-left tile of this specific grid
        # We find which actual tiles belong to this grid to get the anchor
        mask = (inverse_indices == g_idx)
        tx_min = tx[mask].min()
        ty_min = ty[mask].min()
        
        # Adjust anchor to the true grid origin (multiple of grid_size)
        tx0 = (tx_min // grid_size) * grid_size
        ty0 = (ty_min // grid_size) * grid_size

        # Generate the expected grid of tiles
        grid_feats = np.zeros((num_expected, feat_dim), dtype=features.dtype)
        grid_coords = np.zeros((num_expected, 3))
        
        found_count = 0
        for i in range(num_expected):
            # Calculate local offsets: j for rows, i for columns
            local_tx = tx0 + (i % grid_size)
            local_ty = ty0 + (i // grid_size)
            
            grid_coords[i] = [local_tx * tile_size, local_ty * tile_size, group_counter]
            
            # O(1) Dictionary Lookup instead of O(N) DataFrame Scan
            if (local_tx, local_ty) in tile_to_feat_idx:
                grid_feats[i] = features[tile_to_feat_idx[(local_tx, local_ty)]]
                found_count += 1
        
        # 5. Coverage Check
        if (found_count / num_expected) >= min_coverage_thr:
            feats_out.append(grid_feats)
            coords_out.append(grid_coords)
            group_counter += 1

    if not feats_out:
        return np.zeros((0, feat_dim)), np.zeros((0, 3))

    return np.concatenate(feats_out), np.concatenate(coords_out)

def aggregate_tile_tokens(features, coords):
    """
    features: (M, F)
    coords: (M, 3) -> [x, y, grid_id]
    """
    if coords.shape[1] <= 2:
        return features

    # 1. Sort features by Grid ID
    sort_idx = np.argsort(coords[:, 2])
    sorted_features = features[sort_idx]

    # 2. Infer the grid capacity (G^2)
    num_unique_grids = len(np.unique(coords[:, 2]))
    
    # 3. Reshape automatically
    # -1 tells NumPy to "figure out" the width based on N
    return sorted_features.reshape(num_unique_grids, -1)