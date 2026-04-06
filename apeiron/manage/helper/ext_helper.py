from apeiron.utils import save_json, save_img, load_img, deep_get, deep_assign, save_h5_data, load_h5_datas
import numpy as np
import pandas as pd
from pathlib import Path

def get_h5_ext_id(extractions_manifest, target_ext_configs, tgt_coords_configs={}, is_updatable=True):
    """Determine HDF5 extraction ID for given configuration.
    
    Checks if embeddings with matching configuration already exist. Handles
    special cases where patch_tokens can be reused for different strategies.
    
    Args:
        target_ext_configs (dict): Desired extraction configuration:
            - ext_enc: Encoder tile size (e.g., 224)
            - ext_mpp: Microns per pixel (e.g., 0.5)
            - ext_model: Model name (e.g., 'hop0')
            - ext_patch_strategy: 'keep', 'aggr', or 'discard'
        tgt_coords_configs (dict): Tile coordinate generation config:
            - use_mask: Whether tissue masking was used
            - tile_threshold: Tissue ratio threshold
            - stride: Tile overlap
        extractions_manifest (dict): Existing extraction records
    
    Returns:
        tuple: (cur_ext_id, is_generated, updated_manifest)
            - cur_ext_id (str): Extraction ID to use (e.g., 'ext_1')
            - is_generated (bool): True if matching extraction exists
            - updated_manifest (dict): Manifest with new entry if needed
    """
    def pop_key(cfg, key: str):
        if is_updatable:
            return cfg.pop(key, None)  
        else:
            return cfg.get(key)
    
    # Prepare target extraction configs
    tgt_ext_cfg = target_ext_configs.copy()
    tgt_cor_cfg = tgt_coords_configs.copy()
    tgt_ext_strat = pop_key(tgt_ext_cfg, key='ext_patch_strategy')
    tgt_cor_ann = pop_key(tgt_cor_cfg, key='ann_mask')
    
    # Loop all extracted features to find a match
    matched = False
    replace_ext_id = None
    for ext_id, configs in extractions_manifest.items():

        # Prepare pre-extracted configs
        ext_cfg = configs.get('ext_configs', {}).copy()
        cor_cfg = configs.get('coords_configs', {}).copy()

        # Get specific keys to evaluate ext_id
        ext_strat = pop_key(ext_cfg, key='ext_patch_strategy')
        cor_ann = pop_key(cor_cfg, key='ann_mask')

        # Check if match
        if (ext_cfg == tgt_ext_cfg) and (tgt_cor_cfg == cor_cfg):
            
            if is_updatable:
                if tgt_ext_strat == 'keep' and ext_strat not in ['keep']:
                    replace_ext_id = ext_id
                elif tgt_ext_strat == 'aggr' and ext_strat not in ['aggr', 'keep']:
                    replace_ext_id = ext_id

                if tgt_cor_ann == False and cor_ann == True:
                    replace_ext_id = ext_id

            matched = True
            break
    
    # Return instructions for h5
    if not matched or replace_ext_id:
        cur_ext_id = replace_ext_id if replace_ext_id else f'ext_{len(extractions_manifest)+1}'
        deep_assign(extractions_manifest, [cur_ext_id, 'ext_configs'], value=target_ext_configs)
        if tgt_coords_configs:
            deep_assign(extractions_manifest, [cur_ext_id, 'coords_configs'], value=tgt_coords_configs)
        is_generated = False
    else:
        cur_ext_id = ext_id
        is_generated = True
        
    return cur_ext_id, is_generated, extractions_manifest

def split_embeddings_to_chunks(embeddings, chunk_size):
    """Split embedding arrays into fixed-size chunks for HDF5 storage.

    Handles two cases:
    - **Grouped coords** (N, 3): Chunks by unique image_id (3rd column),
      keeping all windows from the same image together.
    - **Flat coords** (N, 2) or None: Chunks by row count.

    Args:
        embeddings (dict): Embedding arrays (``class_token``, ``coords``, etc.).
        chunk_size (int): Maximum number of units (images or rows) per chunk.

    Returns:
        tuple: (chunked_embeddings, total_units) where chunked_embeddings is
            a list of dicts and total_units is the count processed.
    """
    coords = embeddings.get('coords')
    
    # 1. Determine Grouping Strategy
    if coords is not None and coords.shape[1] == 3:
        # Case: (M, 3) where index 2 is the 'image_id'
        image_ids = coords[:, 2]
        unique_ids = np.unique(image_ids)
        total_units = len(unique_ids) # We chunk by Number of Slides
        
        # Determine how many unique IDs fit in a chunk
        num_chunks = int(np.ceil(total_units / chunk_size))
        chunked_embeddings = []

        for i in range(num_chunks):
            start_id_idx = i * chunk_size
            end_id_idx = min((i + 1) * chunk_size, total_units)
            
            # Identify which unique IDs belong to this chunk
            target_ids = unique_ids[start_id_idx:end_id_idx]
            
            # Create a boolean mask for all rows matching these IDs
            mask = np.isin(image_ids, target_ids)
            
            # Slice all keys in the dict using the mask
            chunked_embeddings.append({k: v[mask] for k, v in embeddings.items()})
            
        return chunked_embeddings, total_units # Returns count of IDs processed

    else:
        # Fallback: Original row-based slicing for N,2 or None
        total_len = embeddings['class_token'].shape[0]
        num_chunks = int(np.ceil(total_len / chunk_size))
        chunked_embeddings = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_len)
            chunked_embeddings.append({k: v[start_idx:end_idx] for k, v in embeddings.items()})
            
        return chunked_embeddings, total_len

def update_ext_csv(new_ext_csv, ext_csv_path):
    """Merge updated extraction CSV with the on-disk version.

    Uses ``combine_first`` to fill in newly computed chunk_id and local_id
    values without overwriting existing entries.

    Args:
        new_ext_csv (pd.DataFrame): Updated extraction tracking DataFrame.
        ext_csv_path (str or Path): Path to the on-disk CSV file.
    """
    # 1. Load existing data
    disk_df = pd.read_csv(ext_csv_path).set_index("tile_id")

    # 2. Ensure new_ext_csv is also indexed correctly
    if new_ext_csv.index.name != "tile_id":
        new_ext_csv = new_ext_csv.set_index("tile_id")

    # 3. Use combine_first
    final_df = new_ext_csv.combine_first(disk_df)

    # 4. Write back to CSV
    final_df.reset_index().to_csv(ext_csv_path, index=False)

def initialize_ext_csv(new_tiles_df, ext_id, extractions_csv_path_header):
    """Load or create the extraction tracking CSV for a tile class.

    Merges new tile entries with any existing CSV, deduplicates by tile_id,
    and returns the combined DataFrame indexed by tile_id.

    Args:
        new_tiles_df (pd.DataFrame): New tiles with ``tile_id`` and ``tile_path`` columns.
        ext_id (str): Extraction ID (e.g. ``'ext_1'``).
        extractions_csv_path_header (Path): Base path for the CSV file
            (``{artifact_folder}/{tile_class}``).

    Returns:
        tuple: (extractions_csv, extractions_csv_path) where extractions_csv
            is a DataFrame indexed by tile_id.
    """
    # 3. Load or Initialize CSV
    extractions_csv_path = f"{extractions_csv_path_header}_{ext_id}.csv"
    if Path(extractions_csv_path).exists():
        extractions_csv = pd.read_csv(extractions_csv_path)
    else:
        extractions_csv = pd.DataFrame(columns=["tile_id", "chunk_id", "local_id", "tile_path"])
        extractions_csv.to_csv(extractions_csv_path, index=False)

    # 4. Combine First Logic - ONLY concat relevant columns to avoid extra junk columns
    extractions_csv = pd.concat([extractions_csv, new_tiles_df], ignore_index=True)
    extractions_csv = extractions_csv.drop_duplicates(subset=['tile_id'], keep='first')
    extractions_csv = extractions_csv.set_index("tile_id", drop=True)
    return extractions_csv, extractions_csv_path


def load_tile_embeddings(extractions_csv, ext_h5_path, cur_ext_id):
    """Load tile embeddings from chunked HDF5 using the extraction CSV as an index.

    Reads chunk_id and local_id from the CSV to locate each tile's embeddings
    within the HDF5 file, then reassembles them in tile_id order.

    Args:
        extractions_csv (pd.DataFrame): Extraction tracking CSV with columns
            ``tile_id``, ``chunk_id``, ``local_id``, ``tile_path``.
        ext_h5_path (str or Path): Path to the extraction HDF5 file.
        cur_ext_id (str): Extraction ID within the HDF5 file.

    Returns:
        tuple: (final_tile_ids, final_tile_paths, embeddings) where
            embeddings is a dict of numpy arrays, or None if CSV is empty.
    """
    
    df = extractions_csv.dropna(subset=["chunk_id", "local_id"])
    if df.empty:
        return None

    chunk_ids = df["chunk_id"].astype(int).unique()
    
    # load_h5_datas returns a list of dicts: [{'feature': ...}, {'feature': ...}]
    loaded_chunks, _ = load_h5_datas(ext_h5_path, [[cur_ext_id, cid] for cid in chunk_ids])
    
    # Map them for easy lookup: { chunk_id: data_dict }
    chunk_map = dict(zip(chunk_ids, loaded_chunks))
    
    # vectorized gather
    chunk_keys = next(iter(chunk_map.values())).keys()
    chunked_embeddings_dict = {k: [] for k in chunk_keys}
    final_tile_ids = []
    final_tile_paths = []

    grouped = df.reset_index().sort_values("local_id").groupby("chunk_id")
    last_l_idx = 0
    for c_idx, group in grouped:
        chunk_embeddings = chunk_map[c_idx]

        # coords: (N,C) and C is x, y, window_id
        has_window_id = (chunk_embeddings['coords'].shape[1] == 3)
        if has_window_id:
            ids_in_coords = chunk_embeddings['coords'][:, 2].copy()  # Extract the ID column first
            chunk_embeddings['coords'][:, 2] = chunk_embeddings['coords'][:, 2] + last_l_idx  # Update the coords

        # Now iterate through the tiles inside this specific chunk
        # We use to_numpy() on the group for speed
        for t_idx, l_idx, t_path in group[["tile_id", "local_id", "tile_path"]].to_numpy():

            final_tile_ids.append(t_idx)
            final_tile_paths.append(t_path)

            if has_window_id:
                slice_indices = np.where(np.isin(ids_in_coords, l_idx))[0]  # Check which match your list
            else:
                slice_indices = int(l_idx)

            for k in chunk_keys:
                chunked_embeddings_dict[k].append(chunk_embeddings[k][slice_indices])
        
        last_l_idx += l_idx + 1

    # Output Results that match tile id, path with embeddings
    if has_window_id:
        embeddings = {k: np.concatenate(v, axis=0) for k, v in chunked_embeddings_dict.items()}
    else:
        embeddings = {k: np.stack(v, axis=0) for k, v in chunked_embeddings_dict.items()}

    return final_tile_ids, final_tile_paths, embeddings
