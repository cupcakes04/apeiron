from apeiron.utils import save_json, save_img, load_img, deep_get, deep_assign, save_h5_data, save_h5_datas, load_h5_data, load_h5_datas, update_dict
import numpy as np
from .helper.ext_helper import get_h5_ext_id, split_embeddings_to_chunks, update_ext_csv, initialize_ext_csv, load_tile_embeddings
import pandas as pd
from typing import Literal, Any, Callable

class ArtifactIO:
    """Handles reading and writing of slide/tile artifacts (thumbnails, embeddings, features).

    Provides a structured I/O layer between the Manager's generation pipeline
    and the on-disk artifact storage (PNG images, HDF5 embeddings, JSON manifests).
    Each artifact type has a triplet of methods:

    - ``io_*_in``  — check if the artifact already exists and prepare paths.
    - ``io_*_out`` — save the generated artifact and update the manifest.
    - ``io_*_load`` — load a previously saved artifact from disk.

    Supports two I/O modes:

    - ``'w'`` (write): Generate and save artifacts.
    - ``'r'`` (read): Only load existing artifacts without generating new ones.

    Args:
        **kwargs: Forwarded to parent classes.

    Attributes:
        io_mode (str): Current I/O mode (``'w'`` or ``'r'``).
        read_only (bool): True when ``io_mode='r'``.
        slide_data_paths (dict): Cache of slide artifact paths keyed by slide_id.
        tile_data_paths (dict): Cache of tile artifact paths keyed by tile_class.
        ext_h5_path (str): Path to the current extraction HDF5 file.
        cur_ext_id (str): Current extraction ID within the HDF5 file.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_mode = 'w'
        self.read_only = False
        self.slide_data_paths = {}
        self.tile_data_paths = {}

        self.ext_h5_path = None
        self.cur_ext_id = None

    def set_io_mode(self, mode='r'):
        """Set the I/O mode for artifact operations.

        Args:
            mode ('r' | 'w'): ``'r'`` for read-only (no generation/saving),
                ``'w'`` for write (generate and save). Default ``'r'``.
        """
        if mode in ['r', 'w']:
            self.io_mode = mode
            if mode == 'r':
                self.read_only = True
            if mode == 'w':
                self.read_only = False

    def _is_need_generate(self, is_generated):
        """Check if an artifact needs to be generated.

        Returns True only when in write mode and the artifact does not
        already exist.
        """
        return bool(self.io_mode == 'w' and not is_generated)

    def _is_need_cache(self, is_generated):
        """Check if artifact paths should be cached internally.

        Returns True when in write mode or when the artifact already exists
        (so its path can be used for serving).
        """
        return bool(self.io_mode == 'w' or is_generated)
    
    def _load_slide_reg_data(self, reg_data):
        """Unpack a slide registry entry tuple into instance attributes."""
        (
            self.slide_id, 
            self.slide_name, 
            self.slide_path, 
            self.artifact_folder, 
            self.artifact_manifest_path, 
            self.artifact_manifest
        ) = reg_data
        

### (1): thumbnails
    def io_thumbnail_in(self, reg_data, mode, overview_mpp, method=None):
        """Prepare thumbnail I/O: check existence and set save path.

        Args:
            reg_data (tuple): Slide registry entry from ``slide_entries``.
            mode (str): Thumbnail type (``'slide_thumbnail'`` or ``'masked_thumbnail'``).
            overview_mpp (float): Target microns per pixel.
            method (str, optional): Masking method (for masked_thumbnail only).

        Returns:
            bool: True if a matching thumbnail already exists.
        """
        self._load_slide_reg_data(reg_data)

        self.tb_save_path = f"{self.artifact_folder}/{self.slide_name}_{mode}.png"

        # If thumbnail exists (or same), skip, else we open it
        if mode == 'masked_thumbnail':
            info_dict = {'overview_mpp': overview_mpp, 'method': method}
        else:
            info_dict = {'overview_mpp': overview_mpp}
        is_generated = bool(self.artifact_manifest.get(mode) == info_dict)

        if self._is_need_generate(is_generated):
            self.artifact_manifest[mode] = info_dict
        
        return is_generated

    def io_thumbnail_out(self, is_generated, mode, thumbnail):
        """Save thumbnail to disk and update the artifact manifest.

        Args:
            is_generated (bool): Whether the thumbnail was already generated.
            mode (str): Thumbnail type.
            thumbnail (np.ndarray or None): Generated thumbnail array.
        """
        
        if self._is_need_generate(is_generated):
            save_img(thumbnail, self.tb_save_path)
            save_json(self.artifact_manifest_path, self.artifact_manifest, io_mode=self.io_mode)

        if self._is_need_cache(is_generated):
            self.slide_data_paths[self.slide_id][mode] = self.tb_save_path

    def io_thumbnail_load(self, load_slide_id, mode):
        """Retrieve the cached path for a previously generated thumbnail.

        Args:
            load_slide_id (str): Slide UUID.
            mode (str): Thumbnail type.

        Returns:
            str or None: Path to the thumbnail image file.
        """
        thumbnail_path = self.slide_data_paths[load_slide_id].get(mode)
        return thumbnail_path
        

### (2.1): embeddings (slide)
    def io_embeddings_s_in(self, reg_data, ext_configs, coords_configs, search_slide_annotation: Callable):
        """Prepare slide embedding I/O: determine extraction ID and check existence.

        Args:
            reg_data (tuple): Slide registry entry.
            ext_configs (dict): Extraction configuration (model, encoder, mpp, etc.).
            coords_configs (dict): Tile coordinate generation configuration.
            search_slide_annotation (Callable): Function to find annotation files.

        Returns:
            tuple: (is_generated, coords_configs) where is_generated is True if
                matching embeddings exist, and coords_configs may be updated with
                an annotation path.
        """
        self._load_slide_reg_data(reg_data)
        
        self.ext_h5_path = f"{self.artifact_folder}/{self.slide_name}_extractions.h5"
        
        # If embeddings exists (or same), skip, else we open it
        self.cur_ext_id, is_generated, extractions_manifest = get_h5_ext_id(
            self.artifact_manifest.get('extractions', {}), ext_configs, coords_configs, 
        )
        
        # Extract the embeddings with the given ext_configs
        if self._is_need_generate(is_generated):
            self.artifact_manifest['extractions'] = extractions_manifest

            if coords_configs['ann_mask']:
                ann_path, ann_type = search_slide_annotation(self.slide_name)
                if ann_path:
                    coords_configs = update_dict(coords_configs, ann_mask=ann_path)

        return is_generated, coords_configs
        
    def io_embeddings_s_out(self, is_generated, embeddings):
        """Save slide embeddings to HDF5 and update the artifact manifest.

        Args:
            is_generated (bool): Whether embeddings were already generated.
            embeddings (dict): Embedding arrays to save.
        """

        # Extract the embeddings with the given ext_configs
        if self._is_need_generate(is_generated):

            save_h5_data(
                file_path=self.ext_h5_path, data_dict=embeddings,
                group_parts=['extractions', self.cur_ext_id],
            )
        
            # Assign metadata to artifact_manifest
            save_json(self.artifact_manifest_path, self.artifact_manifest, io_mode=self.io_mode)
        
        # Cache the slide metadata to slide_data_paths
        if self._is_need_cache(is_generated):
            self.slide_data_paths[self.slide_id]['extractions'] = {'ext_h5_path': self.ext_h5_path, 'cur_ext_id': self.cur_ext_id}

    def io_embeddings_s_load(self, load_slide_id):
        """Load slide embeddings from HDF5.

        Args:
            load_slide_id (str): Slide UUID.

        Returns:
            dict or None: Embedding arrays, or None if not available.
        """
        ext_h5_dict = self.slide_data_paths[load_slide_id].get('extractions')
        if ext_h5_dict:
            ext_h5_path, cur_ext_id = ext_h5_dict['ext_h5_path'], ext_h5_dict['cur_ext_id']
            embeddings, _ = load_h5_data(ext_h5_path, group_parts=['extractions', cur_ext_id])
            return embeddings
        else:
            return None

            
### (2.2): embeddings (tile)
    def _load_tile_data_dict(self, tile_data_dict):
        """Unpack a tile entry dictionary into instance attributes."""
        
        self.tile_class = tile_data_dict['tile_class']
        self.new_tiles_df = tile_data_dict['new_tiles_df']
        self.artifact_folder = tile_data_dict['artifact_folder']
        self.artifact_manifest_path = tile_data_dict['artifact_manifest_path']
        self.artifact_manifest = tile_data_dict['artifact_manifest']
        
    def io_embeddings_t_in(self, tile_data_dict, ext_configs):
        """Prepare tile embedding I/O: initialize CSV tracker and find missing tiles.

        Args:
            tile_data_dict (dict): Tile entry from ``tile_entries``.
            ext_configs (dict): Extraction configuration.

        Returns:
            tuple: (is_generated, missing_tile_paths, tile_base_mpp).
        """
        self._load_tile_data_dict(tile_data_dict)

        self.ext_h5_path = f"{self.artifact_folder}/{self.tile_class}_extractions.h5"

        self.cur_ext_id, _, extractions_manifest = get_h5_ext_id(
            self.artifact_manifest.get("extractions", {}), ext_configs, is_updatable=False
        )
        self.extractions_csv, self.extractions_csv_path = initialize_ext_csv(
            self.new_tiles_df, self.cur_ext_id,
            extractions_csv_path_header = (self.artifact_folder / self.tile_class)
        )

        # find missing rows
        self._missing_mask = self.extractions_csv["local_id"].isna()
        missing_tile_paths = self.extractions_csv.loc[self._missing_mask, "tile_path"].values
        
        is_generated = True if missing_tile_paths.size == 0 else False
        if self._is_need_generate(is_generated):
            self.artifact_manifest["extractions"] = extractions_manifest
        tile_base_mpp = self.artifact_manifest["base_mpp"]
        return is_generated, missing_tile_paths, tile_base_mpp
    
        
    def io_embeddings_t_out(self, is_generated, embeddings, chunk_size=10000):
        """Save tile embeddings to chunked HDF5 and update CSV tracker.

        Args:
            is_generated (bool): Whether embeddings were already generated.
            embeddings (dict): Embedding arrays to save.
            chunk_size (int): Number of tiles per HDF5 chunk. Default 10000.
        """
        
        # Extract the embeddings with the given ext_configs
        if self._is_need_generate(is_generated):
            
            # Handle empty column or all-NaN column gracefully
            current_max = self.extractions_csv['chunk_id'].max()
            start_chunk_idx = (current_max + 1) if not pd.isna(current_max) else 0
            chunked_embeddings, total_len = split_embeddings_to_chunks(embeddings, chunk_size)
            
            indices = np.arange(total_len)
            
            # Calculate values
            assigned_chunks = (start_chunk_idx + (indices // chunk_size)).astype(int)
            assigned_locals = (indices % chunk_size).astype(int)

            # Map back to DF (using tile_id as index for speed)
            self.extractions_csv.loc[self._missing_mask, 'chunk_id'] = assigned_chunks.tolist()
            self.extractions_csv.loc[self._missing_mask, 'local_id'] = assigned_locals.tolist()
            
            save_h5_datas(
                file_path=self.ext_h5_path, data_dicts=chunked_embeddings,
                group_parts=[[self.cur_ext_id, ac] for ac in np.unique(assigned_chunks)],
            )
        
            # Assign metadata to artifact_manifest
            save_json(self.artifact_manifest_path, self.artifact_manifest, io_mode=self.io_mode)
            update_ext_csv(self.extractions_csv, self.extractions_csv_path)
            
        # Cache the slide metadata to slide_data_paths
        if self._is_need_cache(is_generated):
            self.tile_data_paths[self.tile_class]['base_mpp'] = self.artifact_manifest["base_mpp"]
            final_tile_ids, final_tile_paths, embeddings = load_tile_embeddings(self.extractions_csv, self.ext_h5_path, self.cur_ext_id)
            ext_data_dict = {"tile_ids": final_tile_ids, "tile_paths": final_tile_paths, "embeddings": embeddings}
            self.tile_data_paths[self.tile_class].update(ext_data_dict)
            

### (3): feats_color
    def io_feats_color_in(self, reg_data, feats_configs):
        """Prepare feature color I/O: check if matching features exist.

        Args:
            reg_data (tuple): Slide registry entry.
            feats_configs (dict): Feature configuration (window_level, patch_to_tile, grid_size).

        Returns:
            bool: True if matching feature colors already exist.
        """
        self._load_slide_reg_data(reg_data)
        
        # If featrues exists (or same), skip, else we open it
        ext_h5_dict = self.slide_data_paths[self.slide_id].get('extractions')
        if ext_h5_dict:
            self.ext_h5_path, self.cur_ext_id = ext_h5_dict['ext_h5_path'], ext_h5_dict['cur_ext_id']
            is_generated = deep_get(self.artifact_manifest, ['extractions', self.cur_ext_id, 'feats_configs']) == feats_configs
        else:
            is_generated = False

        if self._is_need_generate(is_generated):
            deep_assign(self.artifact_manifest, ['extractions', self.cur_ext_id, 'feats_configs'], value=feats_configs)

        return is_generated

    def io_feats_color_out(self, is_generated, data_dict):
        """Save feature colors to HDF5 and update the artifact manifest.

        Args:
            is_generated (bool): Whether features were already generated.
            feats_color (np.ndarray): (N, 3) RGB feature colors.
            feats_size (int): Spatial size of feature windows.
            coords (np.ndarray): (N, 2+) feature coordinates.
        """

        if self._is_need_generate(is_generated):
            save_h5_data(
                file_path=self.ext_h5_path, data_dict=data_dict, 
                group_parts=['extractions', self.cur_ext_id, 'feats_color'],
            )
            
            # Assign metadata to artifact_manifest
            save_json(self.artifact_manifest_path, self.artifact_manifest, io_mode=self.io_mode)

    def io_feats_color_load(self, load_slide_id):
        """Load feature colors from HDF5.

        Args:
            load_slide_id (str): Slide UUID.

        Returns:
            dict: Dictionary with ``feats_color``, ``feats_size``, ``coords``
                keys, or empty dict if not available.
        """
        ext_h5_dict = self.slide_data_paths[load_slide_id].get('extractions')
        if ext_h5_dict:
            ext_h5_path, cur_ext_id = ext_h5_dict['ext_h5_path'], ext_h5_dict['cur_ext_id']
            data_dict, attr = load_h5_data(ext_h5_path, group_parts=['extractions', cur_ext_id, 'feats_color'])
            return data_dict
        else:
            return {}
                
