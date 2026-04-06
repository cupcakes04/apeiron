from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import subprocess
import pyvips
from pathlib import Path
import numpy as np
import json
import pandas as pd

# ------------------------------------- ADDITIONAL HELPERS -------------------------------------

def vips_compress_wsi_batch(wsi_paths: List[str]|str, save_dir: Optional[str] = None, quality: int = 70, 
                            base_mpp: Optional[float] = None, verbose: bool = False):
    """
    Compress multiple WSIs using VIPS.

    Args:
        wsi_paths (List[str]): List of WSI paths to compress, or str to process just one.
        save_dir (Optional[str]): Directory to save compressed WSIs. If None, replace original file.
        quality (int): JPEG quality factor, lower = smaller size (default=80).
        base_mpp (Optional[float]): Base microns-per-pixel value to store as DPI metadata.
        verbose (bool): If False, suppress VIPS output for each file.

    Returns:
        List of Paths to compressed WSIs.
    """
    if isinstance(wsi_paths, (str, Path)):
        wsi_paths = [wsi_paths]
    compressed_paths = []

    for wsi_path in tqdm(wsi_paths, desc="Compressing WSIs"):
        compressed = vips_compress_wsi(wsi_path, save_dir=save_dir, quality=quality, 
                                       base_mpp=base_mpp, verbose=verbose)
        compressed_paths.append(compressed)

    return compressed_paths

def vips_compress_wsi(wsi_path: str | Path, save_dir: Optional[str] = None, quality: int = 70, 
                       base_mpp: Optional[float] = 0.5, verbose: bool = True):
    """Compress a whole slide image using VIPS with JPEG compression and pyramid tiling.
    
    Creates a pyramidal TIFF with JPEG compression for efficient storage and viewing.
    Converts color space to sRGB to fix color issues. Requires libvips installed.
    
    Installation:
        sudo apt install libvips-tools
    
    Args:
        wsi_path (str or Path): Path to input WSI file
        save_dir (str or Path, optional): Output directory. Uses input directory if None
        quality (int): JPEG quality factor (1-100, lower=smaller). Default 70
        base_mpp (float): Microns per pixel for DPI metadata. Default 0.5
        verbose (bool): Print progress messages. Default True

    Returns:
        Path: Path to compressed TIFF file
    
    Note:
        Output filename will have .tiff extension regardless of input format.
        Skips compression if output file already exists.
    """
    wsi_path = Path(wsi_path)

    # Determine output path
    if save_dir is None:
        output_path = wsi_path.parent / (wsi_path.stem + ".tiff")
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / (wsi_path.stem + ".tiff")
    if output_path.is_file():
        if verbose:
            print(f"[INFO] {output_path} has already been created.")
        return output_path
    
    # 1. Load the image
    # autoclean=True helps with memory management
    img = pyvips.Image.new_from_file(str(wsi_path), access="sequential")

    # 2. FIX COLOR: Force transformation to sRGB 
    # This addresses the pink/green issue by standardizing the spectrum
    if img.interpretation != 'srgb':
        print(f"Converting {img.interpretation} to sRGB...")
        img = img.colourspace('srgb')

    # 3. Calculate Resolution (dpm, dots per mm, Pixels/mm)
    # 1000 is the number of pixels in a milimeter
    # `resunit` dictates what tiff.XResolution values are: 2000 dpmm = 50800
    dpmm = int(1000 / base_mpp) 
    tile_dim = 5120
    
    # 4. Save as Pyramid TIFF
    img.tiffsave(
        str(output_path),
        compression="jpeg",
        Q=quality,
        tile=True,
        tile_width=tile_dim,
        tile_height=tile_dim,
        pyramid=True,
        resunit="inch",
        xres=dpmm,
        yres=dpmm,
        bigtiff=True
    )
    return output_path

def sync_and_merge(df_base, df_new, join_on, id_col, subset_cols, output_path):
    """Merge label columns from a new DataFrame into a base registry DataFrame.

    Performs a left join on ``join_on``, appends to any existing CSV at
    ``output_path``, deduplicates by ``id_col`` (keeping the latest), and
    saves the result.

    Args:
        df_base (pd.DataFrame): Base registry DataFrame (e.g. from slide/tile registry).
        df_new (pd.DataFrame): New label DataFrame to merge in.
        join_on (str): Column name to join on (e.g. ``'slide_name'``).
        id_col (str): Column used for deduplication (e.g. ``'slide_id'``).
        subset_cols (list[str]): Label column names to extract from ``df_new``.
        output_path (str or Path): Path to save the merged CSV.

    Returns:
        pd.DataFrame: The merged, deduplicated, and saved DataFrame.
    """
    # 1. Filter df_new first (Stay small!)
    df_new_filtered = df_new[[join_on] + subset_cols].drop_duplicates(subset=join_on)

    # 2. Merge df_new WITH df_base (Left join on df_new)
    # This only processes rows present in your new data
    updated_chunk = pd.merge(df_new_filtered, df_base, on=join_on, how="left")

    # 3. Handle Persistence
    if Path(output_path).exists():
        existing_df = pd.read_csv(output_path)
        # Combine existing data with the NEWLY enriched rows
        final_df = pd.concat([existing_df, updated_chunk], ignore_index=True)
    else:
        final_df = updated_chunk

    # 4. Deduplicate (Keep the newest version of the ID)
    final_df = final_df.drop_duplicates(subset=id_col, keep='last').reset_index(drop=True)
    
    # 5. Save
    final_df.to_csv(output_path, index=False)
    return final_df

def convert_geojson_to_custom(file_path, LABEL_MAP):
    """Convert a GeoJSON annotation file to APEIRON's custom JSON format.

    Reads a GeoJSON file (e.g. from QuPath) and transforms each feature
    into the annotation dictionary format expected by :class:`Annotator`.

    Args:
        file_path (str or Path): Path to the GeoJSON file.
        LABEL_MAP (dict): Mapping of GeoJSON classification names to
            ``{'label': str, 'label_id': int}`` dictionaries.

    Returns:
        dict: Transformed annotation dictionary in APEIRON format.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    transformed = {}
    
    for i, feature in enumerate(data.get("features", [])):
        # Extract original data
        orig_props = feature.get("properties", {})
        class_name = orig_props.get("classification", {}).get("name", "unknown")
        coords = np.array(feature["geometry"]["coordinates"][0]) # Assuming Polygon
        
        # Get mapping config or use a default
        config = LABEL_MAP.get(class_name, {"label": class_name, "label_id": 0})
        
        ann_key = f"ann_{i+1}_name"
        
        entry = {
            "properties": {
                "label": config["label"],
                "label_id": config["label_id"]
            },
            "type": 'polygon',
            "geometry": {}
        }

        # Default to polygon vertices
        entry["geometry"] = {
            "vertices": coords.tolist()
        }
        
        transformed[ann_key] = entry
        
    return transformed