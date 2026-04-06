from typing import Literal
from pathlib import Path
from apeiron.utils import read_json, load_tiff
import numpy as np
import cv2
from matplotlib.path import Path
from shapely.geometry import box, Polygon, Point

class Annotator:
    """Assigns per-tile class labels from shape (JSON) or pixel (mask) annotations.

    Supports two annotation formats:

    - **Shape annotations** (``ann_type='shape'``): JSON dictionaries of polygons
      and ellipses with class labels. Each tile's class fractions are computed
      from the geometric intersection area.
    - **Pixel annotations** (``ann_type='pixel'``): Binary or multi-class mask
      images (uint8 TIFF/PNG). Each tile's class fractions are computed from
      the pixel coverage within the tile area.

    Optionally, ``supervision=True`` groups tiles into annotation-region bags
    for multiple-instance learning.

    Example ``shape_ann_json`` input::

        {
            "ann_1_name": {
                "properties": {"label": "nucleus", "label_id": 4},
                "type": "ellipse",
                "geometry": {"center": [33000.5, 53000.1], "axes": [10300, 13245]}
            },
            "ann_2_name": {
                "properties": {"label": "cell_wall", "label_id": 5},
                "type": "polygon",
                "geometry": {"vertices": [[33000, 51000], [34000, 51000], ...]}
            }
        }

    Args:
        num_classes (int): Total number of classes (including background at index 0).
        **kwargs: Forwarded to parent classes.

    Attributes:
        ann_type (str or None): Active annotation format (``'shape'`` or ``'pixel'``).
        supervision (bool): Whether to produce annotation bags for MIL.
        class_id_map (dict): Mapping of class index to class name.
        num_classes (int): Number of classes.
        pixel_ann_mask (np.ndarray): Loaded pixel annotation mask (H, W) or (H, W, C).
        shape_ann_json (dict): Loaded shape annotation dictionary.
        annotation (np.ndarray): (N, C) class fraction matrix for all tiles.
        objects (list[dict]): List of annotation bags (populated when ``supervision=True``).
    """
    def __init__(self, num_classes: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.ann_type = None
        self.supervision = False
        self.class_id_map: dict = {}
        self.num_classes = num_classes

        self.pixel_ann_mask: np.ndarray = None
        self.shape_ann_json: dict = {}

        self.annotation: np.ndarray = None
        self.objects: list[dict] = []
        
    def load_annotations(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def setup_ann_configs(self, 
        ann_type: Literal['shape', 'pixel'] = None, 
        supervision: bool = False, 
        class_id_map: dict = {},
        background_ratio: float = 0.2,
        **kwargs
    ):
        """Configure annotation settings.

        Args:
            ann_type ('shape' | 'pixel', optional): Annotation format to use.
            supervision (bool): Enable annotation-region bagging for MIL.
            class_id_map (dict): Mapping of class index to class name.
                Sets ``num_classes`` to the length of this map.
        """
        if ann_type:
            self.ann_type = ann_type
        if supervision:
            self.supervision = supervision
        if class_id_map:
            self.class_id_map = class_id_map
            self.num_classes = len(class_id_map)
        if background_ratio:
            self.background_ratio = background_ratio

    def process_annotations(self, coords, tile_size, ann_path, active_coords=False):
        """Load annotations and compute per-tile class fractions.

        Dispatches to ``label_coords_by_json`` or ``label_coords_by_mask``
        depending on the configured ``ann_type``.

        Args:
            coords (np.ndarray): (N, 2+) tile coordinates [x, y, ...].
            tile_size (int): Tile size in pixels at base resolution.
            ann_path (str or Path): Path to the annotation file.
            active_coords (bool): If True, return a boolean activation mask
                instead of class fractions. Default False.
        """
        if ann_path is None:
            return

        if self.ann_type == 'shape':
            self.shape_ann_json = read_json(ann_path)
            self.annotation, self.objects = self.label_coords_by_json(coords, tile_size, active_coords)

        elif self.ann_type == 'pixel':
            self.pixel_ann_mask = load_tiff(ann_path)
            self.annotation = self.label_coords_by_mask(coords, tile_size, active_coords)
            self.objects = None

    def label_coords_by_mask(self, coords, tile_size, active_coords=False):
        """
        Calculates the percentage coverage for each class within the tile area.
        
        Returns:
            pixel_fractions: (N, C) array where each entry is [0.0, 1.0] 
                            representing the density of that class in the tile.
        """
        # 1. Setup scales and mask
        down_scale = self._calc_mask_scales()
        # tile_size_scaled is the size of the patch in pixel-mask units
        # e.g., if patch is 224px and mask is 16x smaller, patch is 14 units in mask
        tile_size_scaled = int(round(tile_size / down_scale))
        mask = self.pixel_ann_mask  # Expected shape (H, W, C)
        N = coords.shape[0]

        # Initialize the output array (N, C)
        pixel_fractions = np.zeros((N, self.num_classes), dtype=np.float32)
        pixel_activations = np.zeros(N, dtype=bool)

        # 2. Iterate through coordinates and calculate density
        for idx, coord in enumerate(coords[:, :2] / down_scale):
            # Convert original slide coordinates to mask coordinates
            x_start = int(coord[0])
            y_start = int(coord[1])
            
            x_end = int(coord[0] + tile_size_scaled)
            y_end = int(coord[1] + tile_size_scaled)

            # 3. Boundaries Check
            y_start = np.clip(y_start, 0, mask.shape[0])
            y_end = np.clip(y_end, 0, mask.shape[0])
            x_start = np.clip(x_start, 0, mask.shape[1])
            x_end = np.clip(x_end, 0, mask.shape[1])

            # 4. Extract Tile Mask and Calculate Mean
            # If the tile area is 0 due to clipping at the very edge, it remains 0.0
            if y_end > y_start and x_end > x_start:
                tile_mask = mask[y_start:y_end, x_start:x_end, :] # (H, W, C)
                if active_coords:
                    pixel_activations[idx] = np.any(tile_mask[..., 1:])
                    continue
                
                # Sum pixels for all classes at once and divide by area
                # We use .mean() over spatial axes or .sum() / area
                pixel_fractions[idx] = tile_mask.mean(axis=(0, 1))

        # Clip values to [0, 1] to handle any rounding/area issues
        pixel_fractions = np.clip(pixel_fractions, 0.0, 1.0)

        if active_coords:
            return pixel_activations
        return pixel_fractions

    def label_coords_by_json(self, coords, tile_size, active_coords=False):
        """
        Returns:
            shape_fractions: (N, C) original density matrix.
        """
        N = coords.shape[0]
        C = self.num_classes

        # INITIALIZE: Everything is 100% Class 0 (Background)
        shape_fractions = np.zeros((N, self.num_classes), dtype=np.float32)
        shape_activations = np.zeros(N, dtype=bool)
        shape_fractions[:, 0] = 1.0
        tile_area = tile_size * tile_size

        # 1. Convert JSON to Shapely and pre-calculate BBoxes
        prepared_shapes = []
        for obj in self.shape_ann_json.values():
            raw_id = obj['properties']['label_id']

            # Initialize a weight vector for this specific shape
            weight_vector = np.zeros(C, dtype=np.float32)
            if isinstance(raw_id, int):
                weight_vector[raw_id] = 1.0
            elif isinstance(raw_id, dict):
                # Expecting: {"ids": [1, 2], "weights": [0.7, 0.3]}
                ids = raw_id.get('ids', [])
                weights = raw_id.get('weights', [])
                for i, w in zip(ids, weights):
                    weight_vector[i] = w

            geom = obj['geometry']
            if obj['type'] == 'polygon':
                s_poly = Polygon(geom['vertices'])
            elif obj['type'] == 'ellipse':
                # Approximate ellipse as a polygon for area intersection
                center = Point(geom['center'])
                # Create a circle and scale it to axes to form ellipse
                circ = center.buffer(1) 
                s_poly = Polygon([(center.x + (p[0]-center.x)*geom['axes'][0], 
                                center.y + (p[1]-center.y)*geom['axes'][1])
                                for p in circ.exterior.coords])
            else:
                continue
            
            # Calculate BBox limits: (minx, miny, maxx, maxy)
            prepared_shapes.append({
                'weight_vector': weight_vector, 
                'poly': s_poly, 
                'bbox': s_poly.bounds 
            })

        # 2. Original Loop: Calculate Class Densities
        for idx, (tx, ty) in enumerate(coords[:, :2]):            
            # Create a bounding box for the current tile
            tile_poly = box(tx, ty, tx + tile_size, ty + tile_size)
            
            for shape in prepared_shapes:

                # Check if tile and shape even touch to save time
                s_poly = shape['poly']
                if tile_poly.intersects(s_poly):
                    if active_coords:
                        shape_activations[idx] = True
                        continue

                    # Calculate the area of intersection
                    intersect_area = tile_poly.intersection(shape['poly']).area
                    fraction = intersect_area / tile_area

                    # Apply the fractional area distributed by the weights
                    # We subtract from background and add to specific labels
                    weights = shape['weight_vector']
                    
                    # If the shape has any weight in non-background classes:
                    # (Assuming index 0 is background)
                    non_bg_weight = 1.0 - weights[0]
                    shape_fractions[idx, :] += fraction * weights
                    shape_fractions[idx, 0] -= fraction * non_bg_weight

        # Ensure we don't exceed 1.0 if shapes overlap
        shape_fractions = np.clip(shape_fractions, 0.0, 1.0)
        row_sums = shape_fractions.sum(axis=1)
        shape_fractions /= row_sums[:, np.newaxis]

        # 3. Handle supervision: Group coords by rectangular bounds
        shape_bags = []
        if not active_coords:
            for shape in prepared_shapes:
                b_minx, b_miny, b_maxx, b_maxy = shape['bbox']
                
                # Find all input coords where the tile (top-left) falls within the BBox
                # Note: We include the tile_size buffer so tiles partially overlapping 
                # the bottom/right edges are caught.
                mask = (coords[:, 0] >= b_minx - tile_size) & (coords[:, 0] <= b_maxx) & \
                    (coords[:, 1] >= b_miny - tile_size) & (coords[:, 1] <= b_maxy)

                indices = np.where(mask)[0]
                if len(indices) > 0:
                    shape_bags.append({
                        "label": shape['weight_vector'].tolist(), # List of C
                        "ids": indices,
                    })
                    
        if active_coords:
            return shape_activations, shape_bags
        return shape_fractions, shape_bags

        
### HELPERS
    @staticmethod
    def mask_to_binary(mask, num_classes):
        """Convert a single-channel class mask to a multi-channel binary mask.

        Args:
            mask (np.ndarray): (H, W) uint8 mask with class indices.
            num_classes (int): Total number of classes.

        Returns:
            np.ndarray: (H, W, num_classes) binary mask where each channel
                is 1 where the pixel belongs to that class.
        """
        # Create an array of shape (num_classes,) -> [0, 1, 2, ... C-1]
        classes = np.arange(num_classes)
        
        # Use broadcasting: (H, W, 1) == (num_classes,) 
        # Result is a (H, W, num_classes) boolean array
        binary_mask = (mask[..., None] == classes).astype(np.uint8)
        return binary_mask

    @staticmethod
    def resize_mask(mask, img_w, img_h, target_scale):
        """Resize a mask to match a target downscale factor.

        Args:
            mask (np.ndarray): (H, W) or (H, W, C) mask to resize.
            img_w (int): Original image width in pixels.
            img_h (int): Original image height in pixels.
            target_scale (float): Desired downscale factor (e.g. 16 means
                the mask will be 1/16th of the original dimensions).

        Returns:
            np.ndarray: Resized mask using nearest-neighbor interpolation.
        """
        # Calculate new size based on original image / target_scale
        new_w = int(round(img_w / target_scale))
        new_h = int(round(img_h / target_scale))
        
        # INTER_NEAREST preserves categorical labels (0, 1, 2...)
        mask = cv2.resize(
            mask, 
            (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST
        )
        return mask

    def _calc_mask_scales(self, target_scale=16, tolerance=0.001):
        """Validate and normalize the pixel annotation mask scale.

        Checks whether the mask dimensions match the expected downscale
        factor relative to the slide dimensions. Resizes the mask if the
        scale deviates beyond ``tolerance``, and converts a single-channel
        mask to multi-channel binary format if needed.

        Args:
            target_scale (int): Expected downscale factor. Default 16.
            tolerance (float): Maximum acceptable relative difference. Default 0.001.

        Returns:
            int: The effective downscale factor after any resizing.
        """
        img_h, img_w = self.dimensions_hw
        mask_h, mask_w = self.pixel_ann_mask.shape[:2]

        # 1. Calculate the current scale
        # We use float division to check for precision
        current_scale_h = img_h / mask_h
        current_scale_w = img_w / mask_w

        # 2. Check for dimension mismatch (relative difference)
        # This ensures the aspect ratio isn't heavily distorted
        rel_diff = abs(current_scale_h - current_scale_w) / max(current_scale_h, current_scale_w)
        
        # 3. Check if we need to resize: 
        # - If scale is not exactly target_scale
        # - Or if the H and W scales are significantly different
        needs_resize = (abs(current_scale_h - target_scale) > tolerance or 
                        rel_diff > tolerance)

        if needs_resize:
            print(f"Resizing mask: Current scale {current_scale_h:.2f}x != Target {target_scale}x")
            self.pixel_ann_mask = self.resize_mask(self.pixel_ann_mask, img_w, img_h, target_scale)

            # After resizing, the scale is exactly the target_scale
            down_scale = target_scale
        else:
            # If within tolerance, use the actual calculated scale
            down_scale = int(round(current_scale_h))
            
        if self.pixel_ann_mask.ndim == 2:
            self.pixel_ann_mask = self.mask_to_binary(self.pixel_ann_mask, self.num_classes)
        
        return down_scale
