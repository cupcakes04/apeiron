from .manage import Manager, Collector
from apeiron.model import Backbone, Inferencer
from typing import Literal, List
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from apeiron.utils import sync_and_merge

class Operator:
    """High-level operator interface for managing slide and tile analysis workflows.

    Extends Collector (-> Manager -> Registry + ArtifactIO) to provide a
    simplified, top-level API for:
    - Ingesting slide/tile data into the registry
    - Generating thumbnails, embeddings, and feature colors in batch
    - Serving pre-configured Analyzer instances for interactive exploration
    - Collecting features as generators for downstream training
    - Creating labeled datasets by merging registry data with external labels

    One Operator is typically instantiated per user session. The Backbone is
    shared across sessions.

    Args:
        **kwargs: Keyword arguments forwarded to Collector / Manager, including:
            - bacbone (Backbone): Shared model backbone for feature extraction.
            - root_dir (str): Database root directory containing SLIDE_DATABASE/ and TILE_DATABASE/.
            - project_path (str): Project directory containing config.yaml and dataset CSVs.
    """
    def __init__(self, backbone: Backbone, root_dir: str, **kwargs):
        self.collector = Collector(backbone=backbone, root_dir=root_dir, device=kwargs.get('device'))
    
    def setup(self, project_path: str):
        """Fully reset the operator to a new project.
        Clears all collected data and reloads configuration from the given
        project directory.

        Args:
            project_path (str): Path to the new project directory.
        """
        self.collector.setup_configs(project_path=project_path)
        self.collector.reset_collector()

### Helpers

    # Registry
    def ingest_data(self, *args, **kwargs):
        return self.collector.ingest_data(*args, **kwargs)

    # Collector
    def query_slide_dataset(self, *args, **kwargs):
        return self.collector.query_slide_dataset(*args, **kwargs)
    def query_tile_dataset(self, *args, **kwargs):
        return self.collector.query_tile_dataset(*args, **kwargs)
    def set_io_mode(self, *args, **kwargs):
        return self.collector.set_io_mode(*args, **kwargs)
    def lookup_table(self, *args, **kwargs):
        return self.collector.lookup_table(*args, **kwargs)

    def generate_thumbnails(self, *args, **kwargs):
        return self.collector.generate_thumbnails(*args, **kwargs)
    def generate_embeddings_slide(self, *args, **kwargs):
        return self.collector.generate_embeddings_slide(*args, **kwargs)
    def generate_embeddings_tile(self, *args, **kwargs):
        return self.collector.generate_embeddings_tile(*args, **kwargs)
    def generate_feats_color(self, *args, **kwargs):
        return self.collector.generate_feats_color(*args, **kwargs)

    def serve_slide_analyzer(self, *args, **kwargs):
        return self.collector.serve_slide_analyzer(*args, **kwargs)
    def serve_tile_analyzer(self, *args, **kwargs):
        return self.collector.serve_tile_analyzer(*args, **kwargs)

    def slide_features_collector(self, *args, **kwargs):
        return self.collector.slide_features_collector(*args, **kwargs)
    def tile_features_collector(self, *args, **kwargs):
        return self.collector.tile_features_collector(*args, **kwargs)
    def intitalise_inferencer(self, *args, **kwargs):
        return self.collector.intitalise_inferencer(*args, **kwargs)
    def similarity_search(self, *args, **kwargs):
        return self.collector.similarity_search(*args, **kwargs)

    # Inference
    def train(self, *args, **kwargs):
        return self.collector.train(*args, **kwargs)
    def evaluate(self, *args, **kwargs):
        return self.collector.evaluate(*args, **kwargs)
    def val_graphs(self, *args, **kwargs):
        return self.collector.analyzer.val_graphs(*args, **kwargs)
    def plot_history(self, *args, **kwargs):
        return self.collector.plot_history(*args, **kwargs)

    # Additional
    def create_datasets(self, labels_path, label_col: List|str, mode: Literal['slide', 'tile'], output_path=None):
        """Create a labeled dataset CSV by merging external labels with registry data.

        replcae the name of data 
        - slide -> `slide_name`
        - tile -> `tile_name`

        replace the label col with `LABEL`

        Reads a label CSV from the project directory, joins it with the
        appropriate registry (slide or tile) on the name column, deduplicates
        by ID, and persists the result.

        Args:
            labels_path (str): Relative path (from project_path) to the label CSV.
            label_col (str or list[str]): Column name(s) in the label CSV to merge.
            mode ('slide' | 'tile'): Which registry to merge against.
            output_path (str or Path, optional): Output CSV path. Defaults to
                ``project_path / '{mode}_dataset.csv'``.

        Returns:
            pd.DataFrame: The merged and deduplicated dataset.
        """
        # 1. Configuration based on mode
        if mode == 'slide':
            df_reg = self.collector.slide_reg.df
            join_on = 'slide_name'
            id_col = 'slide_id'
            default_name = 'slide_dataset.csv'
        elif mode == 'tile':
            df_reg = self.collector.tile_reg.df
            join_on = 'tile_name'
            id_col = 'tile_id'
            default_name = 'tile_dataset.csv'

        # 2. Setup Paths
        if output_path is None:
            output_path = self.collector.project_path / default_name
        
        full_labels_path = self.collector.project_path / labels_path
        df_labels = pd.read_csv(full_labels_path)

        # 3. Standardize label_col to a list
        subset_cols = [label_col] if isinstance(label_col, str) else label_col

        # 4. Use Helper
        return sync_and_merge(
            df_base=df_reg,
            df_new=df_labels,
            join_on=join_on,
            id_col=id_col,
            subset_cols=subset_cols,
            output_path=output_path
        )