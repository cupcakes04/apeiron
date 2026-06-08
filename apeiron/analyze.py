from .model import Backbone, Extractor, Inferencer
from .entity import Processor, SlideTiles, StandaloneTiles, WindowTiles
from typing import Literal
import numpy as np
from functools import partial

class Analyzer(Extractor, Processor, Inferencer):
    """Main analyzer for whole slide image (WSI) and tile-based analysis.

    Combines feature extraction (via Extractor) with post-processing, annotation,
    and visualization (via Processor) to provide a single-entity analysis interface.
    Each Analyzer instance operates on one slide or tile set at a time, while the
    Backbone is shared across all Analyzer instances.

    Inheritance chain:
        Analyzer -> Extractor  (embedding extraction)
                 -> Processor  -> Reader     (slide/tile I/O, tile coordinate generation)
                               -> Annotator  (shape/pixel annotation labeling)
                               -> Visualiser (overlay creation and blending)

    Args:
        backbone (Backbone): Shared foundational model manager for feature extraction.
        **kwargs: Forwarded to parent classes (e.g. ``ext_enc``, ``ext_mpp``).

    Attributes:
        backbone (Backbone): The shared model backbone instance.
        model_name (str): Name of the currently assigned foundational model.
        model (FmWrappers): Wrapped neural network used for forward passes.
        transform (FmTransform): Image preprocessing pipeline for the active model.
        device (torch.device): Computation device (CPU/GPU).
    """
    def __init__(self, backbone: Backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.mdatas: list = []

    # |-----------------------------------------------|
    # |-------------------- Open ---------------------|
    # |-----------------------------------------------|

    def reset_extractions(self):
        """Clear both embeddings and processed features to free memory."""
        self.reset_embeddings()
        self.reset_features()

    def open_slide(self, slide_path, base_mpp=None):
        """Open a whole slide image for analysis.
        
        Args:
            slide_path (str or Path): Path to the slide file (e.g., .svs, .tiff, .ndpi)
            base_mpp (float, optional): Base microns per pixel. If None, inferred from metadata
        """
        self.reset_extractions()
        self.update_res_unit(ext_mpp=base_mpp)
        self.setup_slide(slide_path, base_mpp)
        
    def open_tiles(self, tile_paths, base_mpp=None):
        """Open a set of tile images for analysis.

        Args:
            tile_paths (list[str | Path]): List of file paths to tile images.
            base_mpp (float, optional): Base microns per pixel for the tiles.
                Required for windowed extraction; ignored for standalone.
        """
        self.reset_extractions()
        self.update_res_unit(ext_mpp=base_mpp)
        self.setup_tiles(tile_paths, base_mpp)
        
    def assign_model(self, model_name='hop0'):
        """Assign a foundational model from the backbone for feature extraction.
        
        Args:
            model_name (str): Name of the foundational model. Options include:
                - 'hop0': H-optimus-0
                - 'hop1': H-optimus-1
                - 'vir1': Virchow
                - 'vir2': Virchow2
                - 'ch15': CONCH 1.5
                - 'uni2h': UNI2-h
                - 'mstar': mSTAR
                - 'dino3': DINOv3
        """
        self.backbone.select_model(model_name)
        self.model_feats_dim = self.backbone.model_feats_dim
        self.model_name = self.backbone.model_name
        self.model = self.backbone.model
        self.transform = self.backbone.transform
        self.model_dim = self.model_feats_dim[self.model_name]

        
    # |-----------------------------------------------|
    # |------------ Prepare extractions --------------|
    # |-----------------------------------------------|
        

    def prepare_tiles_dataset(self, mode: Literal['slide', 'standalone', 'windowed'] = None):
        """Prepare a PyTorch dataset from slide tiles for batch feature extraction.
        
        Creates a SlideTiles dataset using the generated tile coordinates,
        which can be used with a DataLoader for efficient batch processing.
        
        Sets:
            self.dataset: SlideTiles dataset instance ready for DataLoader
        """
        if mode == 'slide':
            self.extract_dataset = SlideTiles(self.slide_path, self.tile_coords, self.encoder, transform=self.transform)

        if mode == 'standalone':
            self.extract_dataset = StandaloneTiles(self.tile_paths, transform=self.transform)
                
        if mode == 'windowed':
            self.extract_dataset = WindowTiles(self.tile_paths, self.tile_coords, transform=self.transform)
        
    def prepare_features(self, 
                         window_level: Literal["grid", "tile", "patch"] = 'tile', 
                         patch_to_tile: Literal["max", "mean", "discard"] = 'mean', 
                         grid_size: int = 2):
        """Prepare features from embeddings for visualization and analysis.
        
        Processes raw embeddings into features at different spatial resolutions.
        
        Args:
            window_level (str): Spatial resolution level for feature analysis:
                - 'tile': Default window size matching the encoder (e.g., 224x224)
                - 'grid': Merge multiple tiles into larger grids (e.g., 2x2 tiles)
                - 'patch': Use individual patch tokens for fine-grained segmentation
            patch_to_tile (str): How to aggregate patch tokens into tile-level features:
                - 'max': Use maximum pooling across patches
                - 'mean': Use average pooling across patches
                - 'discard': Only use class token, ignore patch tokens
            grid_size (int): Number of tiles per dimension when window_level='grid' (e.g., 2 for 2x2)
        """
        self.reset_features()
        self.assign_features(self.embeddings, window_level, patch_to_tile, grid_size)

    def prepare_annotations(self, ann_path=None, ann_type=None):
        """Load and apply annotations to the current feature coordinates.

        Configures the annotation type and computes per-tile class fractions
        or activations from the provided annotation file.

        Args:
            ann_path (str or Path): Path to annotation file (.json for shape, .tiff/.png for pixel).
            ann_type (str): Annotation format — ``'shape'`` (JSON polygons/ellipses) or
                ``'pixel'`` (binary/multi-class mask image).
        """
        self.setup_ann_configs(ann_type=ann_type)
        self.process_annotations(self.proc_ext.coords, self.proc_ext.feats_size, ann_path=ann_path)
        

    # |-----------------------------------------------|
    # |----------------- Downstream ------------------|
    # |-----------------------------------------------|


    def prepare_inferencer(self,
        mode=None, feats_configs=None, inf_models = 'abmil', 
        lbl_class_id_map=None, ann_class_id_map=None,
        lbl_cls_weights=None, ann_cls_weights=None,
        lbl_loss_type='bce', ann_loss_type='bce',
        lr=1e-4, optimizer='adam', weight_decay: float = 0.0, scheduler: str = None,
        return_cfgs=False,
    ):
        # If feature aggregation, feature size * 2
        in_features = self.model_dim
        is_patch = bool(feats_configs['window_level'] == 'patch')
        if feats_configs['patch_to_tile'] in ['max', 'mean'] and not is_patch:
            in_features = in_features * 2
        
        configs = {
            'ext_enc': 224, 'ext_mpp': 0.5, 'ext_model': self.model_name, 
            'feats_configs': feats_configs, 'in_features': in_features, 'mode': mode,
            'lbl_n_classes': len(lbl_class_id_map or {}), 'ann_n_classes': len(ann_class_id_map or {}),
            'lbl_class_id_map': lbl_class_id_map, 'ann_class_id_map': ann_class_id_map,
            'lbl_cls_weights': lbl_cls_weights, 'ann_cls_weights': ann_cls_weights,
            'inf_models': inf_models, 'lbl_loss_type': lbl_loss_type, 'ann_loss_type': ann_loss_type,
            'lr': lr, 'optimizer': optimizer, 'weight_decay': weight_decay, 'scheduler': scheduler,
        }
        if return_cfgs:
            return configs
        else:
            self.setup_inferencer(**configs)

    def predict(self, 
        mode: Literal['slide', 'tile'], 
        feats_configs=None, ann_path=None, ann_type=None, 
        ground_truth=False, collect_ids=None, batch_size=300):
        """Run downstream inference on the currently loaded slide or tiles.
        
        Applies the configured `Inferencer` to the extracted features. Handles data batching, 
        prediction, and post-processing into a consolidated `ModelData` structure.
        
        Args:
            mode (str): 'slide' for whole-slide processing, 'tile' for discrete tiles.
            feats_configs (dict, optional): Dict to re-prepare features before prediction.
            ann_path (str, optional): Path to annotations to load for ground truth comparison.
            ann_type (str, optional): Type of annotation ('shape' or 'pixel').
            ground_truth (bool): Whether to include loaded labels/annotations in the batch for metric calculation.
            collect_ids (list, optional): Specific tile indices to process (for 'tile' mode).
            batch_size (int): Number of tiles/samples per batch during inference.
            
        Returns:
            ModelData: Consolidated dataclass containing predictions (pred_lbl, pred_ann, etc.).
        """
        
        if feats_configs:
            self.prepare_features(**feats_configs)
        if ann_path:
            self.prepare_annotations(ann_path, ann_type)

        if mode == 'tile':
            predata = self.tile_preprocessor(indices=collect_ids)
        if mode == 'slide': 
            predata = self.slide_preprocessor(ground_truth=ground_truth)

        mdatas, datas = [], []
        for data in predata.generator(batch_size=batch_size, shuffle=False):
            pred = self.predict_data(data, run_metric=bool(ground_truth))
            mdatas.append(pred['mdata'])
            datas.append(pred['data'])
        
        self.mdatas = mdatas
        return self.postprocessor(mdatas, datas)

    def store_inferencer(self, store_path):
        self.save_stored_inference(store_path)
