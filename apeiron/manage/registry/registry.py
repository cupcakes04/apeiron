from apeiron.utils import mkdir, read_json, save_h5_data, load_h5_data
from pathlib import Path
import pandas as pd
import uuid
import os
from typing import List, Literal
from .slide_reg import SlideRegistry
from .tile_reg import TileRegistry

class Registry:
    """Unified registry managing both slide and tile databases.

    Delegates to ``SlideRegistry`` and ``TileRegistry`` for their respective
    data types.  Provides a single entry point for data ingestion and
    generator-based iteration over registry entries with artifact metadata.

    Database Layout::

        root_dir/
        ├── SLIDE_DATABASE/
        │   ├── DATASETS/      # Slide files organized by class
        │   ├── ARTIFACTS/     # Generated slide artifacts
        │   └── registry.csv   # Slide metadata
        └── TILE_DATABASE/
            ├── DATASETS/      # Tile images organized by class
            ├── ARTIFACTS/     # Generated tile artifacts
            └── registry.csv   # Tile metadata

    Args:
        root_dir (str): Root directory containing SLIDE_DATABASE/ and TILE_DATABASE/.
        **kwargs: Forwarded to parent classes.

    Attributes:
        slide_reg (SlideRegistry): Slide registry instance.
        tile_reg (TileRegistry): Tile registry instance.
    """
    def __init__(self, root_dir: str, **kwargs):
        super().__init__(**kwargs)
        root_dir = Path(root_dir)

        self.slide_reg = SlideRegistry(root_dir=root_dir / "SLIDE_DATABASE")
        self.tile_reg = TileRegistry(root_dir=root_dir / "TILE_DATABASE")

    def ingest_data(self, data_classes, base_mpps=None, mode: Literal['slide', 'tile'] = None):
        """Scan and register data from specified class directories.

        Args:
            data_classes (str or list[str]): Folder name(s) under DATASETS/ to scan.
            base_mpps (float or list[float], optional): Base microns per pixel
                for each tile class (tile mode only).
            mode ('slide' | 'tile'): Which registry to ingest into.
        """

        if mode =='slide':
            self.slide_reg.ingest_slide_classes(data_classes)
        if mode =='tile':
            self.tile_reg.ingest_tile_classes(data_classes, base_mpps)

    def slide_entry_generator(self, slide_ids):
        """Generate registry entries with artifact information for given slide IDs.
        
        Args:
            slide_ids (list): List of slide UUIDs to process

        Yields:
            tuple: (slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest)
                - slide_name (str): Filename of the slide
                - slide_path (Path): Full path to slide file
                - artifact_folder (Path): Directory for slide artifacts
                - artifact_manifest_path (str): Path to manifest JSON
                - artifact_manifest (dict): Loaded manifest or empty dict
        """
        
        # Iterate each slide
        rows = self.slide_reg.query_registry(slide_ids=slide_ids)
        for _, row in rows.iterrows():
            
            slide_name = row["slide_name"]
            slide_id = row["slide_id"]
            slide_path = row["slide_path"]
            artifact_folder = self.slide_reg.artifacts_dir / f"slide_{slide_id}"
            mkdir(artifact_folder)
            
            slide_path = self.slide_reg.datasets_dir / slide_path
            artifact_folder = self.slide_reg.artifacts_dir / artifact_folder
            artifact_manifest_path = f"{artifact_folder / slide_name}.json"
            artifact_manifest = read_json(artifact_manifest_path)
            
            yield slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest
            
            
    def tile_entry_generator(self, tile_ids: list):
        """Generate registry entries with artifact information for given tile IDs.

        Groups tiles by tile_class and yields one entry per class containing
        the tile DataFrame, artifact folder, manifest path, and manifest.

        Args:
            tile_ids (list): List of tile UUIDs to process.

        Yields:
            dict: Tile entry with keys ``tile_class``, ``new_tiles_df``,
                ``artifact_folder``, ``artifact_manifest_path``, ``artifact_manifest``.
        """
        rows = self.tile_reg.query_registry(tile_ids)
        base_str = str(self.tile_reg.datasets_dir)
            
        for tile_class, frame in rows.groupby("tile_class"):
            # 1. Prepare frame paths 
            frame["tile_path"] = frame["tile_path"].apply(lambda p: f"{base_str}/{p}")
            
            # 2. Setup paths and folder
            artifact_folder = self.tile_reg.artifacts_dir / f"tile_{tile_class}"
            artifact_manifest_path = f"{artifact_folder / tile_class}.json"
            artifact_folder.mkdir(parents=True, exist_ok=True)
            
            new_tiles_df = frame[["tile_id", "tile_path"]]
            artifact_manifest = read_json(artifact_manifest_path)
            artifact_manifest['base_mpp'] = self.tile_reg._get_tile_class_base_mpp(Path(base_str) / tile_class)
            yield {
                'tile_class': tile_class,
                'new_tiles_df': new_tiles_df,
                'artifact_folder': artifact_folder,
                'artifact_manifest_path': artifact_manifest_path,
                'artifact_manifest': artifact_manifest,
            }
