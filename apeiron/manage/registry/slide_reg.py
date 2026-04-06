from pathlib import Path
import pandas as pd
import uuid
import os
from typing import List
from apeiron.utils import mkdir, read_json
from apeiron.utils import convert_to_list


# Temporary Registry instead of postsqlgret thing, so i just leave all func here
# TODO: Implement sql postgre stuff


SLIDE_EXTENSIONS = {".svs", ".ndpi", ".png", ".tif", ".tiff", ".jpeg", ".jpg"}

class SlideRegistry:
    """Manages slide registry and artifact storage for the database.
    
    Maintains a CSV registry mapping slide IDs to file paths and metadata.
    Handles slide ingestion, artifact directory management, and registry queries.
    
    Database Structure:
        root_dir/
        ├── DATASETS/          # Original slide files organized by class
        ├── ARTIFACTS/         # Generated data (thumbnails, embeddings, etc.)
        └── registry.csv       # Slide metadata registry
    
    Args:
        root_dir (str): Root directory of the database
    
    Attributes:
        root_dir (Path): Database root directory
        datasets_dir (Path): Directory containing original slide files
        artifacts_dir (Path): Directory for generated artifacts
        registry_path (Path): Path to registry.csv file
        df (pd.DataFrame): Registry dataframe with columns:
            - slide_id: Unique UUID for each slide
            - slide_name: Original filename
            - slide_path: Relative path from DATASETS/
            - slide_class: Category/folder name
    """
    def __init__(self, root_dir: str, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.datasets_dir = self.root_dir / "DATASETS"
        self.artifacts_dir = self.root_dir / "ARTIFACTS"
        self.registry_path = self.root_dir / "registry.csv"
        self.df: pd.DataFrame

        if self.registry_path.exists():
            self.df = pd.read_csv(self.registry_path)
        else:
            self.df = pd.DataFrame(
                columns=["slide_id", "slide_name", "slide_path", "slide_class"]
            )
            

    def _scan_folder(self, folder: Path, extensions: list):
        """Recursively scan folder for valid slide files.
        
        Args:
            folder (Path): Directory to scan
        
        Yields:
            tuple: (relative_path, filename) for each valid slide file
        """
        for root, _, files in os.walk(folder):
            for fname in files:
                fname = Path(fname)
                ext = fname.suffix.lower()
                if ext in extensions:
                    full_path = (Path(root) / fname).relative_to(self.datasets_dir)
                    yield full_path, fname.stem

    def _save_registry(self):
        self.df.to_csv(self.registry_path, index=False)

    def _register_slide(self, slide_path: Path, slide_name: str, slide_class: str):
        """Register a new slide in the registry with unique UUID.
        
        Skips if slide path already exists in registry to avoid duplicates.
        
        Args:
            slide_path (Path): Relative path to slide file from DATASETS/
            slide_name (str): Filename of the slide
            slide_class (str): Category/class name for the slide
        """
        slide_path = str(slide_path)

        # Already registered?
        if slide_path in self.df["slide_path"].values:
            return

        slide_id = uuid.uuid4().hex

        row = {
            "slide_id": slide_id,
            "slide_name": slide_name,
            "slide_path": slide_path,
            "slide_class": slide_class,
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        
    def ingest_slide_classes(self, slide_classes):
        """Scan and register all slides from specified class directories.
        
        Recursively scans DATASETS/{class}/ folders for valid slide files
        and adds them to the registry with unique UUIDs.
        
        Args:
            slide_classes (str or list): Class name(s) to ingest (e.g., ['colon', 'breast'])
        
        Example:
            registry.ingest_slide_classes(['colon', 'prostate', 'samples'])
        """
        slide_classes = convert_to_list(slide_classes)
        for slide_class in slide_classes:
            slide_class_dir = self.datasets_dir / slide_class
            if not slide_class_dir.exists():
                continue

            for path, name in self._scan_folder(slide_class_dir, SLIDE_EXTENSIONS):
                self._register_slide(path, name, slide_class)
                
            self._save_registry()

        
    def query_registry(self, slide_ids: list[str]):
        """Query registry for slides matching given IDs.
        
        Creates artifact directories if they don't exist.
        
        Args:
            slide_ids (list): List of slide UUIDs to query
        
        Returns:
            dict: Mapping of slide_name to registry information:
                - slide_id: UUID
                - slide_path: Relative path from DATASETS/
                - artifact_path: Full path to artifact directory
        
        Raises:
            KeyError: If no matching slides found
        """
        rows = self.df[self.df["slide_id"].isin(slide_ids)]
        
        if rows.empty:
            raise KeyError("No matching slides found")

        return rows
        