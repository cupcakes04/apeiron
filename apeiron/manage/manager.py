from pathlib import Path
from apeiron.utils import read_csv
from apeiron.analyze import Analyzer
from apeiron.model import Backbone
from .registry import Registry
from .artifact import ArtifactIO
import yaml
import pandas as pd
import numpy as np
from typing import Literal, List
from tqdm import tqdm
from PIL import Image
from apeiron.utils import search_dir, convert_to_list, match_file

class Manager(Registry, ArtifactIO):
    """Manages slide/tile analysis workflows, data generation, and artifact storage.

    Coordinates between the Registry (slide + tile), ArtifactIO, Analyzer, and
    Backbone to generate and cache slide/tile artifacts (thumbnails, embeddings,
    feature colors). Handles project configuration loading, dataset querying,
    and batch generation pipelines.

    Args:
        backbone (Backbone): Shared model backbone for feature extraction.
        project_path (str): Path to project directory containing ``config.yaml``,
            ``slide_dataset.csv``, and ``tile_dataset.csv``.
        **kwargs: Forwarded to Registry (requires ``root_dir``).

    Attributes:
        config (dict): Loaded project configuration from ``config.yaml``.
        slide_dataset (pd.DataFrame): Full slide dataset from ``slide_dataset.csv``.
        tile_dataset (pd.DataFrame): Full tile dataset from ``tile_dataset.csv``.
        selected_slide_dataset (pd.DataFrame): Filtered subset for slide processing.
        selected_tile_dataset (pd.DataFrame): Filtered subset for tile processing.
        analyzer (Analyzer): Analyzer instance configured with project settings.
        backbone (Backbone): Shared model backbone.
        slide_data_paths (dict): Cache of generated artifact paths per slide_id.
        tile_data_paths (dict): Cache of generated artifact paths per tile_class.
    """

    
    # |-----------------------------------------------|
    # |--------------- CONFIGURATIONS ----------------|
    # |-----------------------------------------------|


    def __init__(self, backbone: Backbone, device: Literal['cuda', 'cpu'] = None, **kwargs):
        super().__init__(**kwargs)

        self.config: dict
        self.slide_dataset: pd.DataFrame
        self.tile_dataset: pd.DataFrame
        self.selected_slide_dataset: pd.DataFrame
        self.selected_tile_dataset: pd.DataFrame

        self.device = device
        self.analyzer: Analyzer
        self.backbone = backbone
    
    def setup_configs(self, project_path: str):
        """Load project configuration and slide_dataset from specified directory.
        
        Reads config.yaml for extraction/visualization settings and slide_dataset.csv
        for slide information. Initializes analyzer with configuration parameters.
        
        Args:
            project_path (str): Path to project directory
        """
        
        # Reset/Inititalise these for further processing
        self.cur_data = None
        self.slide_data_paths = {}
        self.tile_data_paths = {}

        # Project paths
        self.project_path = Path(project_path)
        self.config_path = self.project_path / 'config.yaml'
        self.slide_dataset_path = self.project_path / 'slide_dataset.csv'
        self.tile_dataset_path = self.project_path / 'tile_dataset.csv'
        self.stain_norm_path = self.project_path / 'normalise_targets'
        self.annotations_folder = self.project_path / 'annotations'
        self.inferencer_folder = self.project_path / 'inferencer'
        
        # Load configs & dataset
        with open(Path(self.config_path), 'r') as f:
            self.config = yaml.safe_load(f)
        self.slide_dataset = read_csv(self.slide_dataset_path)
        self.tile_dataset = read_csv(self.tile_dataset_path)
        self.available_modes = {'slide': self.slide_dataset is not None, 'tile': self.tile_dataset is not None}
        
        # Assign configs
        self.overview_mpp = self.config.get("overview_mpp", 8.0)
        self.norm_configs = self.config.get("norm_configs", {})
        self.norm_configs['target_path'] = self.stain_norm_path

        # (1): Extractions

        ## Slide
        self.slide_configs = self.config.get('slide_ext', {})
        self.slide_ext_configs = self.slide_configs.get('ext_configs')
        self.slide_coords_configs = self.slide_configs.get('coords_configs')
        self.slide_feats_configs = self.slide_configs.get('feats_configs')
        
        ## Tile
        self.tile_configs = self.config.get('tile_ext', {})
        self.tile_ext_configs = self.tile_configs.get('ext_configs')
        self.tile_coords_configs = self.tile_configs.get('coords_configs')
        self.tile_feats_configs = self.tile_configs.get('feats_configs')

        # (2): Ground Truth
        self.tile_gt_configs = self.config.get('tile_gt', {})
        self.slide_gt_configs = self.config.get('slide_gt', {})
        self.slide_ann_configs = self.slide_gt_configs.get('ann_configs')

        # (3): Downstream & Training
        self.slide_downstream_configs = self.config.get('slide_downstream')
        self.tile_downstream_configs = self.config.get('tile_downstream')

        # Start query & prepare
        self.analyzer = Analyzer(
            self.backbone, device=self.device,
            ext_enc=self.slide_ext_configs['ext_enc'], ext_mpp=self.slide_ext_configs['ext_mpp']
        )
        self.analyzer.load_norm_configs(self.norm_configs)
        self.analyzer.setup_ann_configs(**self.slide_ann_configs)
        self.query_slide_dataset(select_all=True)
        self.query_tile_dataset(select_all=True)
        
                
    # |-----------------------------------------------|
    # |--------------- Query & Search ----------------|
    # |-----------------------------------------------|
        

    def query_slide_dataset(self, slide_ids: list = None, select_all=True):
        """Query and filter the slide dataset to select specific slides for processing.

        Args:
            slide_ids (list, optional): List of slide UUIDs to select.
                Ignored if ``select_all=True``.
            select_all (bool): If True, select all slides in the dataset. Default True.

        Returns:
            pd.DataFrame: Filtered dataset containing selected slides.
        """
        if not self.available_modes['slide']: return
        if select_all:
            self.selected_slide_dataset = self.slide_dataset
        else:
            slide_ids = convert_to_list(slide_ids)
            self.selected_slide_dataset = self.slide_dataset[
                self.slide_dataset["slide_id"].isin(slide_ids)
            ]
        return self.selected_slide_dataset

    def query_tile_dataset(self, tile_ids: list = None, select_all=True):
        """Query and filter the tile dataset to select specific tiles for processing.

        Args:
            tile_ids (list, optional): List of tile UUIDs to select.
                Ignored if ``select_all=True``.
            select_all (bool): If True, select all tiles in the dataset. Default True.

        Returns:
            pd.DataFrame: Filtered dataset containing selected tiles.
        """
        if not self.available_modes['tile']: return
        if select_all:
            self.selected_tile_dataset = self.tile_dataset
        else:
            slide_ids = convert_to_list(slide_ids)
            self.selected_tile_dataset = self.tile_dataset[
                self.tile_dataset["tile_id"].isin(tile_ids)
            ]
        return self.selected_tile_dataset


    def lookup_table(self, input_list: str|list, mode: Literal['slide', 'tile'], to_id=True):
        """Convert between slide/tile names and UUIDs.

        Args:
            input_list (str or list): Name(s) or UUID(s) to look up.
            mode ('slide' | 'tile'): Which registry to query.
            to_id (bool): If True, convert names -> UUIDs.
                If False, convert UUIDs -> names. Default True.

        Returns:
            list: Matching UUIDs (if ``to_id=True``) or names.
        """
        if not self.available_modes[mode]: return
        input_list = convert_to_list(input_list)

        if mode == 'slide':
            df = self.slide_dataset
            d_name = 'slide_name'
            d_id = 'slide_id'
        elif mode == 'tile':
            df = self.tile_dataset
            d_name = 'tile_name'
            d_id = 'tile_id'

        if to_id:
            return df[df[d_name].isin(input_list)][d_id].tolist()
        else:
            return df[df[d_id].isin(input_list)][d_name].tolist()
        
    def search_slide_annotation(self, slide_name):
        """Search for annotation files matching a slide name.

        Looks in the project's ``annotations/`` directory for shape (JSON)
        or pixel (TIFF/PNG) annotation files whose stem matches the slide
        name. Falls back to the other type if the preferred type is not found.

        Args:
            slide_name (str): Slide filename stem to search for.

        Returns:
            tuple: (ann_path, ann_type) where ann_path is a Path or None,
                and ann_type is ``'shape'`` or ``'pixel'``.
        """
        if not self.available_modes['slide']: return

        # Find Annotation Files (uint8 mask or JSON geometry)
        ann_path = None
        ann_type = self.slide_ann_configs['ann_type']
        if ann_type == 'shape':
            ann_path = match_file((self.annotations_folder / 'shape'), slide_name, extensions={'.json', '.geojson'})
            if not ann_path:
                ann_type = 'pixel'
                ann_path = match_file((self.annotations_folder / 'pixel'), slide_name, extensions={'.tiff', '.tif', '.png', '.jpg', '.jpeg'})

        elif ann_type == 'pixel':
            ann_path = match_file((self.annotations_folder / 'pixel'), slide_name, extensions={'.tiff', '.tif', '.png', '.jpg', '.jpeg'})
            if not ann_path:
                ann_type = 'shape'
                ann_path = match_file((self.annotations_folder / 'shape'), slide_name, extensions={'.json', '.geojson'})

        return ann_path, ann_type
                
                
    # |-----------------------------------------------|
    # |----------------- GENERATION ------------------|
    # |-----------------------------------------------|


    @property
    def slide_entries(self):
        """Generator yielding slide SlideRegistry entries for selected dataset.
        
        Queries the SlideRegistry using slide_ids from selected_slide_dataset and caches
        slide paths for later use.
        
        Yields:
            tuple: (slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest)
        """
        slide_ids = self.selected_slide_dataset["slide_id"]
        for reg in self.slide_entry_generator(slide_ids):
            slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest = reg
            self.slide_data_paths.setdefault(slide_id, {})
            self.slide_data_paths[slide_id]['slide_path'] = str(slide_path)
            yield slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest

    @property
    def tile_entries(self):
        """Generator yielding tile registry entries for the selected tile dataset.

        Queries the TileRegistry using tile_ids from ``selected_tile_dataset``,
        groups by tile_class, and caches tile data paths.

        Yields:
            dict: Tile entry with keys ``tile_class``, ``new_tiles_df``,
                ``artifact_folder``, ``artifact_manifest_path``, ``artifact_manifest``.
        """
        tile_ids = self.selected_tile_dataset["tile_id"]
        for results in self.tile_entry_generator(tile_ids):
            tile_class = results['tile_class']
            self.tile_data_paths.setdefault(tile_class, {})
            yield results
    
    def generate_thumbnails(self, overview_mpp=None, modes: List[str] = ["slide_thumbnail", "masked_thumbnail"]):
        """Generate and save slide thumbnails at specified resolution.
        
        Creates various thumbnail types for visualization and tissue detection.
        Skips generation if thumbnails already exist with matching parameters.
        
        Args:
            overview_mpp (float, optional): Microns per pixel for thumbnails. Uses config value if None
            modes (list): Types of thumbnails to generate:
                - 'slide_thumbnail': Original RGB thumbnail
                - 'masked_thumbnail': Binary tissue mask
        """
        if not self.available_modes['slide']: return
        input_overview_mpp = overview_mpp or self.overview_mpp
        
        # Iterate each slide
        for reg_data in tqdm(self.slide_entries):
            slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest = reg_data

            for mode in modes:
                method = self.slide_coords_configs['method']
                is_generated = self.io_thumbnail_in(reg_data, mode, input_overview_mpp, method)

                # Generate thumbnail according to mode
                thumbnail = None
                if self._is_need_generate(is_generated):
                    self.analyzer.open_slide(slide_path)
                    thumbnail = self.analyzer.serve_thumbnail(mode=mode, target_mpp=input_overview_mpp, method=method)
                self.io_thumbnail_out(is_generated, mode, thumbnail)
            
    def generate_embeddings_slide(self, batch_size=300, num_workers=4):
        """Extract and save tile embeddings for all selected slides.
        
        Generates tile coordinates from tissue masks, extracts features using the
        foundational model, and saves embeddings to HDF5 files. Skips slides with
        existing embeddings matching current configuration.
        
        Args:
            batch_size (int): Number of tiles to process per batch. Default 300
            num_workers (int): Parallel workers for data loading. Default 4
        """
        if not self.available_modes['slide']: return
        self.analyzer.assign_model(model_name=self.slide_ext_configs['ext_model'])    
            
        # Iterate each slide
        for reg_data in tqdm(self.slide_entries):
            slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest = reg_data
            is_generated, slide_coords_configs = self.io_embeddings_s_in(
                reg_data, self.slide_ext_configs, self.slide_coords_configs, self.search_slide_annotation
            )
            
            # Extract the embeddings with the given slide_ext_configs
            if self._is_need_generate(is_generated):
                self.analyzer.open_slide(slide_path)
                self.assign_analyzer(slide_id, "masked_thumbnail")
                self.analyzer.create_tile_coords(**slide_coords_configs)
                self.analyzer.prepare_tiles_dataset(mode='slide')
                self.analyzer.extract_tiles(batch_size=batch_size, num_workers=num_workers, 
                                            ext_patch_strategy=self.slide_ext_configs['ext_patch_strategy'])
            self.io_embeddings_s_out(is_generated, self.analyzer.embeddings)

        # Clear memory
        self.analyzer.reset_extractions()
            
    def generate_embeddings_tile(self, batch_size=300, num_workers=4, chunk_size=10000):
        """Extract and save tile embeddings for all selected slides.
        
        Generates tile coordinates from tissue masks, extracts features using the
        foundational model, and saves embeddings to HDF5 files. Skips slides with
        existing embeddings matching current configuration.
        
        Args:
            batch_size (int): Number of tiles to process per batch. Default 300
            num_workers (int): Parallel workers for data loading. Default 4
        """
        if not self.available_modes['tile']: return
        self.analyzer.assign_model(model_name=self.tile_ext_configs['ext_model'])
        
        # Iterate each tile
        for shard_data_dict in tqdm(self.tile_entries):
            is_generated, tile_paths, tile_base_mpp = self.io_embeddings_t_in(shard_data_dict, self.tile_ext_configs)

            # Extract the embeddings with the given tile_ext_configs
            if self._is_need_generate(is_generated):
                self.analyzer.open_tiles(tile_paths, tile_base_mpp)
                self.analyzer.create_windowed_tile_coords(self.tile_coords_configs['stride'])
                self.analyzer.prepare_tiles_dataset(mode=self.tile_ext_configs['ext_type'] )
                self.analyzer.extract_tiles(batch_size=batch_size, num_workers=num_workers, 
                                            ext_patch_strategy=self.tile_ext_configs['ext_patch_strategy'])
            self.io_embeddings_t_out(is_generated, self.analyzer.embeddings, chunk_size)
        
        # Clear memory
        self.analyzer.reset_extractions()
            
    def generate_feats_color(self):
        """Generate and save color-reduced features for visualization.
        
        Applies dimensionality reduction (PCA/UMAP) to embeddings to create
        3D RGB features for overlay visualization. Saves to HDF5 alongside embeddings.
        """
        if not self.available_modes['slide']: return
        # Iterate each slide
        for reg_data in tqdm(self.slide_entries):
            slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest = reg_data
            
            is_generated = self.io_feats_color_in(reg_data, self.slide_feats_configs)
            if self._is_need_generate(is_generated):

                # Compute slide features
                self.analyzer.open_slide(slide_path)
                self.assign_analyzer(slide_id, "embeddings")
                self.analyzer.compute_feats_color(method='pca')

            self.io_feats_color_out(
                is_generated, self.analyzer.proc_ext.get_feat_color_data()
            )


    # |-----------------------------------------------|
    # |-------------------- Assigns ------------------|
    # |-----------------------------------------------|


    def assign_analyzer(self, slide_id, data_modes: List[str]):
        """Load generated data artifacts into the analyzer instance.
        
        Retrieves previously generated data from disk and assigns to analyzer
        for immediate use without regeneration.

        Args:
            slide_name (str): Name of the slide to load data for
            data_modes (list or str): Types of data to load:
                - 'slide_thumbnail': Original slide thumbnail
                - 'masked_thumbnail': Tissue-masked thumbnail
                - 'embeddings': Tile embeddings and coordinates
                - 'feats_color': Pre-computed visualization features
                = 'annotation': Read shape or pixel annotations

                - 'req': Load all essential data types
                - 'ann': Load to support annotation
                - 'pred': Load to support prediction
                - 'all': Load all available data types
        """
        data_modes = convert_to_list(data_modes)
    
        # Assign/generate if None
        if any(k in data_modes for k in ['slide_thumbnail', 'req', 'ann', 'pred', 'all']):
            thumbnail_path = self.io_thumbnail_load(slide_id, 'slide_thumbnail')
            self.analyzer.serve_thumbnail('slide_thumbnail', thumbnail_path, self.overview_mpp)
            
        if any(k in data_modes for k in ['masked_thumbnail', 'req', 'ann', 'pred', 'all']):
            thumbnail_path = self.io_thumbnail_load(slide_id, 'masked_thumbnail')
            self.analyzer.serve_thumbnail('masked_thumbnail', thumbnail_path, self.overview_mpp)

        # Assign only
        if any(k in data_modes for k in ['embeddings', 'req', 'ann', 'pred', 'all']):
            self.analyzer.embeddings = self.io_embeddings_s_load(slide_id)
            self.analyzer.prepare_features(**self.slide_feats_configs)

        if any(k in data_modes for k in ['feats_color', 'req', 'ann', 'all']):
            data_dict = self.io_feats_color_load(slide_id)
            self.analyzer.load_proc_ext(**data_dict)

        # Inferencer & Ground Truth
        if any(k in data_modes for k in ['annotation', 'ann', 'pred', 'all']):
            slide_name = self.lookup_table(slide_id, mode='slide', to_id=False)[0]
            ann_path, ann_type = self.search_slide_annotation(slide_name)
            self.analyzer.prepare_annotations(ann_path, ann_type)
            
        if any(k in data_modes for k in ['prediction', 'pred', 'all']):
            self.analyzer.predict(mode='slide')
            
            
    def serve_slide_analyzer(self, slide_id, data_modes: List[str] = 'req'):
        """Serve a pre-configured analyzer instance with loaded data for a specific slide.
        
        Opens the slide and loads previously generated data (thumbnails, embeddings, features)
        into the analyzer. Avoids redundant reopening if the slide is already loaded.
        
        Args:
            slide_id (str): UUID of the slide from the dataset.
            data_modes (list or str): Types of data to load into the analyzer:
                - 'slide_thumbnail': Original slide thumbnail
                - 'masked_thumbnail': Tissue-masked thumbnail
                - 'embeddings': Extracted tile embeddings and coordinates
                - 'feats_color': Pre-computed color features for visualization
                - 'req': Load all required generated data (default)
                - 'all': Load all available generated data (default)
        
        Returns:
            Analyzer: Configured analyzer instance with loaded data ready for analysis
        """
        if not self.available_modes['slide']: return
        # Avoid reopening and reseting openned analyzer
        if slide_id != self.cur_data:
            self.analyzer.open_slide(self.slide_data_paths[slide_id]['slide_path'])
            self.cur_data = slide_id
        if data_modes:
            self.assign_analyzer(slide_id, data_modes)
        return self.analyzer
        
    def serve_tile_analyzer(self, tile_class):
        """Serve a pre-configured analyzer for a tile class.

        Opens the tile paths and loads cached embeddings into the analyzer.
        Avoids redundant reopening if the tile class is already loaded.

        Args:
            tile_class (str): Tile class name (folder name in the database).

        Returns:
            Analyzer: Configured analyzer with loaded tile embeddings.
        """
        if not self.available_modes['tile']: return

        # Avoid reopening and reseting openned analyzer
        if tile_class != self.cur_data:
            self.analyzer.open_tiles(self.tile_data_paths[tile_class]['tile_paths'], 
                                     base_mpp=self.tile_data_paths[tile_class]['base_mpp'])
            self.cur_data = tile_class

        # Only 1 purpose which is embeddings
        self.analyzer.embeddings = self.tile_data_paths[tile_class]['embeddings']
        self.analyzer.prepare_features(**self.tile_feats_configs)
        return self.analyzer
        
            