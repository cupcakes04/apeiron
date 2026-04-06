from pathlib import Path
import pandas as pd
import uuid
import os
import numpy as np
from typing import List, Generator, Tuple
from apeiron.utils import mkdir, read_json, save_json, save_h5_datas, load_h5_data
from PIL import Image
from tqdm import tqdm
from apeiron.utils import convert_to_list

DEFAULT_BASE_MPP = 0.5

# Temporary Registry instead of postsqlgret thing, so i just leave all func here
# TODO: Implement sql postgre stuff


TILE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpeg", ".jpg"}

class TileRegistry:
    """Manages tile image registry and artifact storage.

    Maintains a CSV registry mapping tile UUIDs to file paths and metadata.
    Handles tile ingestion from class directories and stores base MPP metadata.

    Database Structure::

        root_dir/
        ├── DATASETS/          # Tile images organized by class folder
        ├── ARTIFACTS/         # Generated tile artifacts (HDF5, CSVs, manifests)
        └── registry.csv       # Tile metadata registry

    Args:
        root_dir (str): Root directory of the tile database.
        **kwargs: Forwarded to parent classes.

    Attributes:
        root_dir (Path): Tile database root directory.
        datasets_dir (Path): Directory containing tile image folders.
        artifacts_dir (Path): Directory for generated tile artifacts.
        registry_path (Path): Path to ``registry.csv``.
        df (pd.DataFrame): Registry DataFrame with columns:
            ``tile_id``, ``tile_name``, ``tile_path``, ``tile_class``.
    """
    def __init__(self, root_dir: str, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.datasets_dir = self.root_dir / "DATASETS"
        self.artifacts_dir = self.root_dir / "ARTIFACTS"
        self.registry_path = self.root_dir / "registry.csv"
        
        mkdir(self.artifacts_dir)
        
        self.df: pd.DataFrame

        if self.registry_path.exists():
            self.df = pd.read_csv(self.registry_path)
        else:
            self.df = pd.DataFrame(
                columns=["tile_id", "tile_name", "tile_path", "tile_class"]
            )

    def _scan_folder(self, folder: Path, extensions: list):
        """Recursively scan a folder for valid tile image files.

        Args:
            folder (Path): Directory to scan.
            extensions (set): Valid file extensions (e.g. ``{'.png', '.jpeg'}``).

        Returns:
            tuple: (full_paths, fnames) — lists of relative paths and stems.
        """
        full_paths, fnames = [], []
        for root, _, files in os.walk(folder):
            for fname in files:
                fname = Path(fname)
                ext = fname.suffix.lower()
                if ext in extensions:
                    full_path = (Path(root) / fname).relative_to(self.datasets_dir)
                    full_paths.append(full_path)
                    fnames.append(fname.stem)
        return full_paths, fnames
            
    def _save_registries(self):
        self.df.to_csv(self.registry_path, index=False)

    def _register_tiles(self, tile_names: list[str], tile_paths: list[str], tile_class: str):
        """Register multiple tiles at once, skipping duplicates.

        Args:
            tile_names (list[str]): Tile filename stems.
            tile_paths (list[str]): Relative paths from DATASETS/.
            tile_class (str): Class/folder name for these tiles.

        Returns:
            pd.DataFrame or None: DataFrame of newly registered tiles,
                or None if all tiles were already registered.
        """
        # 1. Filter out tiles that are already in the registry to avoid duplicates
        existing_paths = set(self.df["tile_path"].values)
        
        new_rows = []
        for tile_name, tile_path in zip(tile_names, tile_paths):
            tile_path = str(tile_path)
            if tile_path not in existing_paths:
                # Generate ID and build the row
                row = {
                    "tile_id": uuid.uuid4().hex,
                    "tile_name": tile_name,
                    "tile_path": tile_path,
                    "tile_class": tile_class,
                }
                new_rows.append(row)
                # Add to set so we don't add the same path twice within the same batch
                existing_paths.add(tile_path)

        # 2. Bulk append if there's anything new
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            return new_df
        
        
    def ingest_tile_classes(self, tile_classes, base_mpps=None):
        """Scan tile images and create HDF5 shards with chunked storage.
        
        Recursively scans DATASETS/{class}/ folders, loads images, and stores them
        in HDF5 shards with N images per chunk.
        """
        tile_classes = convert_to_list(tile_classes)

        base_mpps = self._process_input_base_mpps(base_mpps, len(tile_classes))

        for tile_class, base_mpp in zip(tile_classes, base_mpps):
            
            # Collect all image paths for this class
            tile_class_dir = self.datasets_dir / tile_class
            self._set_tile_class_base_mpp(tile_class_dir, base_mpp)
            tile_paths, tile_names = self._scan_folder(tile_class_dir, TILE_EXTENSIONS)
            new_rows = self._register_tiles(tile_names, tile_paths, tile_class)
            if new_rows is None:
                continue
        
            # Save registries
            self._save_registries()

    def query_registry(self, tile_ids: List[str]):
        """Query registry for tiles matching given IDs.

        Args:
            tile_ids (list[str]): List of tile UUIDs to query.

        Returns:
            pd.DataFrame: Matching rows from the registry.

        Raises:
            KeyError: If no matching tiles are found.
        """
        rows = self.df[self.df["tile_id"].isin(tile_ids)]

        if rows.empty:
            raise KeyError("No matching tiles found")
        
        return rows

    # Some base mpp helpers
    @staticmethod
    def _process_input_base_mpps(base_mpps, length):
        """Normalize base_mpps input to a list matching the number of tile classes.

        Args:
            base_mpps: Single value, list, or None. None defaults to 0.5.
            length (int): Expected number of tile classes.

        Returns:
            list[float]: List of base MPP values, one per tile class.
        """
            
        if not base_mpps:
            base_mpps = [DEFAULT_BASE_MPP] * length
        else:
            base_mpps = convert_to_list(base_mpps)
        final_base_mpps = [base_mpp if base_mpp else DEFAULT_BASE_MPP for base_mpp in base_mpps]

        assert len(final_base_mpps) == length
        return final_base_mpps

    @staticmethod
    def _set_tile_class_base_mpp(tile_class_dir, base_mpp):
        """Write the base MPP value to a text file in the tile class directory."""
        with open(Path(tile_class_dir) / "base_mpp.txt", "w") as f:
            f.write(str(base_mpp))

    @staticmethod
    def _get_tile_class_base_mpp(tile_class_dir):
        """Read the base MPP value from a tile class directory."""
        with open(Path(tile_class_dir) / "base_mpp.txt", "r") as f:
            return float(f.read().strip())



    