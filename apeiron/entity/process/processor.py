from ..reader import Reader
from .visualise import Visualiser
from .annotate import Annotator
from .helper.tokens import *
from .helper.preprocess import *
from .helper.features import *
from typing import Literal
import numpy as np
import time
import matplotlib.pyplot as plt
from apeiron.utils import np_unsqueeze
from apeiron.model.inference import ModelData
from typing import List, Literal

class Processor(Reader, Annotator, Visualiser):
    """Processes slide embeddings into features for visualization and analysis.

    Central processing hub that converts raw embeddings into analysis-ready
    features at multiple spatial scales (patch, tile, grid). Provides
    dimensionality reduction, clustering, similarity scoring, and delegates
    overlay rendering to Visualiser and annotation labeling to Annotator.

    Inheritance:
        Processor -> Reader     (slide/tile I/O, coordinate generation)
                  -> Annotator  (shape/pixel annotation labeling)
                  -> Visualiser (overlay creation and blending)

    Args:
        **kwargs: Forwarded to all parent classes.

    Attributes:
        proc_ext (ProcessedExt): Dataclass containing coords, features, and clustering info.
        mdata (ModelData): Dataclass containing model predictions, losses, and metrics.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proc_ext = ProcessedExt()
        self.mdata = ModelData()
        
    def reset_features(self):
        """Clear all feature-related data to free memory."""
        self.proc_ext.reset()
        
    def load_proc_ext(self, **kwargs):
        self.proc_ext.assign(**kwargs)

                
    def assign_features(self, embeddings, 
                        window_level: Literal["grid", "tile", "patch"] = 'tile', 
                        patch_to_tile: Literal["max", "mean", "discard"] = 'mean',
                        grid_size: int = 2,
                        ):
        """Convert embeddings to features at specified spatial resolution.
        
        Processes raw embeddings into features suitable for visualization and analysis.
        Supports multi-scale analysis from fine-grained patches to coarse grids.
        
        Args:
            embeddings (dict): Extracted embeddings containing:
                - 'class_token': (N, F) tile-level features
                - 'patch_tokens': (N, 256, F) patch-level features (optional)
            window_level (str): Spatial scale for features:
                - 'patch': Individual patches (~14x14 pixels each)
                - 'tile': Full tiles (encoder size, e.g., 224x224)
                - 'grid': Groups of tiles (e.g., 2x2 = 448x448)
            patch_to_tile (str): How to use patch tokens:
                - 'discard': Only use class tokens
                - 'max': Concatenate class token with max-pooled patches
                - 'mean': Concatenate class token with mean-pooled patches
            grid_size (int): Tiles per dimension when window_level='grid'
        """
        
        # (A): Grid is a group of tiles, such as 3x3 tiles as a window
        if window_level == "grid":
            tile_features = aggr_patch_into_tile(embeddings, patch_to_tile)
            features, coords = group_class_tokens(tile_features, embeddings['coords'], self.encoder, grid_size)
            feats_size = int(self.encoder * grid_size)
            coords_size = int(self.encoder)
        
        # (B): Tile is default, following the encoder as window, may aggregate patch token
        elif window_level == "tile":
            features = aggr_patch_into_tile(embeddings, patch_to_tile)
            coords = embeddings['coords']
            feats_size = coords_size = int(self.encoder)
            
        # (C): Patch is segmented from tile, such as patch size of 14
        elif window_level == "patch":
            patch_dim, features, coords = flatten_patch_tokens(self.encoder, embeddings['patch_tokens'], embeddings['coords'])
            feats_size = coords_size = int(np.ceil(self.encoder / patch_dim))
        
        # Make sure feats and coords are standardized in dtype
        self.proc_ext = ProcessedExt(
            features=features.astype(np.float32), coords=coords.astype(np.float32), 
            coords_size=coords_size, feats_size=feats_size
        )

    def compute_feats_color(self,
                            method: Literal["pca", "umap"] = 'pca',
                            n_clusters=10,
                            query_indices: int | list[int] = None):
        """Compute visualization-ready feature representations.

        Applies one or more of the following operations on the current features:

        - **Dimensionality reduction** (``method``): Reduce (N, F) features to
          (N, 3) RGB via PCA or UMAP.  Cached to avoid redundant computation.
        - **Clustering** (``n_clusters``): K-means clustering on the reduced
          features, producing cluster indices and a color map.
        - **Similarity scoring** (``query_indices``): Rank all tiles by
          distance to the centroid of the selected query tiles.

        Args:
            method ('pca' | 'umap', optional): Dimensionality reduction method.
                Only recomputed if the method changes from the previous call.
            n_clusters (int, optional): Number of K-means clusters. Set to
                ``None`` or ``0`` to skip clustering.
            query_indices (int or list[int], optional): Tile index(es) to use
                as the query region for similarity scoring.
        """ 
        feats_color = self.proc_ext.feats_color
        
        if feats_color is None:
            _feats = aggregate_tile_tokens(self.proc_ext.features, self.proc_ext.coords)  # if window=grid
            feats_color = reduce_features(_feats, dims=3, nns=50, md=0.1, method=method)
            self.proc_ext.assign(feats_color=feats_color)

        if query_indices:
            feats_score = rank_feats(feats_color, query_indices, scoring_mode='rank', sigma=None)
            self.proc_ext.assign(feats_score=feats_score)
            
        if n_clusters:
            feats_clusters, feats_color_map = cluster_feats(feats_color, n_clusters)
            self.proc_ext.assign(feats_clusters=feats_clusters, feats_color_map=feats_color_map)


    # |-----------------------------------------------|
    # |---------------- Preprocessors ----------------|
    # |-----------------------------------------------|

        
    def slide_preprocessor(self, ground_truth=True) -> PreData:
        coords = self.proc_ext.coords
        features = self.proc_ext.features

        if self.annotation is None:
            ground_truth = False

        iterate = True
        batch_size = None
        
        if self.supervision and ground_truth and self.objects:
            data_type = 'group'
            batch_size = 1  # inhomogenous N
            cord_list, feats_list, ann_list, label_list = bag_data_features(
                coords, features, self.annotation, self.objects)
            pre_data = PreData(coords=cord_list, features=feats_list, annotation=ann_list, label=label_list)

        elif is_coord_a_group(coords):
            data_type = 'group'
            cord_list, feats_list, ann_list = ungroup_data_features(coords, features, self.annotation)
            if ground_truth:
                label_list = [np.mean(ann, axis=0) for ann in ann_list]
                pre_data = PreData(coords=cord_list, features=feats_list, annotation=ann_list, label=label_list)
                pre_data.filter(self.background_ratio)
            else:
                pre_data = PreData(coords=cord_list, features=feats_list)

        else:
            data_type = 'single'
            iterate = False
            pre_data = PreData(
                coords=np_unsqueeze(coords), features=np_unsqueeze(features), 
                annotation=np_unsqueeze(self.annotation), objects=[self.objects]
            )
        
        pre_data.fix_generator_args(iterate=iterate, batch_size=batch_size)
        pre_data.assign_metadata(coords_size=self.proc_ext.coords_size, data_type=data_type)
        return pre_data
        
        
    def tile_preprocessor(self, indices=None) -> PreData:
        coords = self.proc_ext.coords
        features = self.proc_ext.features

        if indices is not None and len(indices) > 0:
            ext_mask = np.isin(coords[:, 2], indices) if is_coord_a_group(coords) else indices
        else:
            ext_mask = np.ones(len(coords), dtype=bool)

        if is_coord_a_group(coords):
            data_type = 'group'
            cord_list, feats_list, _ = ungroup_data_features(coords[ext_mask], features[ext_mask])
            pre_data = PreData(coords=cord_list, features=feats_list)

        else:
            data_type = 'single'
            pre_data = PreData(coords=coords[ext_mask], features=features[ext_mask]) 
            
        pre_data.fix_generator_args(iterate=True)
        pre_data.assign_metadata(coords_size=self.proc_ext.coords_size, data_type=data_type)
        return pre_data


    def postprocessor(self, mdatas: List[ModelData], datas: List[dict]):
        """Consolidates batch-wise model predictions and data into a single structure.
        
        Takes multiple output ModelData items from batched inference and merges 
        them back into a coherent single ModelData structure aligned with the 
        original spatial coordinates.
        
        Args:
            mdatas (List[ModelData]): List of ModelData instances from predictions.
            datas (List[dict]): List of dictionaries containing coordinates and metadata.
            
        Returns:
            ModelData: A unified ModelData instance with post-processed predictions.
        """
        all_crd, all_ann, all_atn, all_lbl, all_obj = [], [], [], [], []
        global_offset_lbl = 0
        global_offset_obj = 0

        for mdata, data_dict in zip(mdatas, datas):

            # 1. Base data
            coords = data_dict['coords']    # (B, K, 2) or (1, N, 2)
            data_type = data_dict.get('data_type', 'single') # 'group' or 'single'
            B, K = coords.shape[:2]

            # 2. Predictions
            ## (a) Label
            if mdata.pred.pred_lbl is not None:
                all_lbl.append(mdata.pred.pred_lbl) # (B, C)

                # Do this if the data runs abmil on each group
                expanded = np.empty((B, K, 3))
                expanded[..., :2] = coords[..., :2]
                expanded[..., 2] = (np.arange(B) + global_offset_lbl)[:, None]
                coords = expanded
                global_offset_lbl += B

            ## (b) Attention
            if mdata.pred.pred_atn is not None:
                pred_atn = mdata.pred.pred_atn # (B, K)
                all_atn.append(pred_atn.reshape(-1))

            ## (c) Annotation
            if mdata.pred.pred_ann is not None:
                pred_ann = mdata.pred.pred_ann # (B, K, C)
                all_ann.append(pred_ann.reshape(-1, pred_ann.shape[-1]))

            ## (d) Objects
            if mdata.pred.pred_obj is not None:
                for b in range(B):
                    for obj in mdata.pred.pred_obj[b]:   # list of length B
                        all_obj.append({
                            'ids': np.array(obj['ids']) + global_offset_obj,
                            'labels': obj['labels'],
                            'scores': obj.get('scores', 0.0)
                        })
                    global_offset_obj += K
            else:
                # Increment offset evenly if no objects
                global_offset_obj += B * K
                
            # Flatten coordinates: (B, K, 2) -> (B*K, 2)
            all_crd.append(coords.reshape(-1, coords.shape[-1]))

        # 3. Final Stacking
        new_mdata = ModelData()
        new_mdata.assign(mode='pred', post_processed=True)
        new_mdata.assign(mode='pred', pred_crd=np.concatenate(all_crd, axis=0) if all_crd else None)
        new_mdata.assign(mode='pred', pred_ann=np.concatenate(all_ann, axis=0) if all_ann else None)
        new_mdata.assign(mode='pred', pred_atn=np.concatenate(all_atn, axis=0) if all_atn else None)
        new_mdata.assign(mode='pred', pred_obj=all_obj if all_obj else None)
        new_mdata.assign(mode='pred', pred_data_type=data_type)

        # labels can be either for a full slide or tiles
        if all_lbl:
            stacked_lbl = np.concatenate(all_lbl, axis=0)  # (Total_B, C)
            if data_type == 'single' and stacked_lbl.shape[0] == 1:
                pred_lbl = stacked_lbl.squeeze(0) # (C,)
            else:
                pred_lbl = stacked_lbl # (N_groups, C)
        else:
            pred_lbl = None
        new_mdata.assign(mode='pred', pred_lbl=pred_lbl)

        self.mdata = new_mdata
        return mdata