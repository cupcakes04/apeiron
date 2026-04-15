from dataclasses import dataclass, field
import numpy as np
import torch
from .tokens import aggregate_patch_tokens

@dataclass
class PreData:
    '''
    objects (list[dict]): Annotation bags with ``'label'`` (list) and ``'ids'`` (np.ndarray of indices).
    '''
    # The Data we want to get
    coords: np.ndarray | list
    features: np.ndarray | list
    annotation: np.ndarray | list | None = None
    label: np.ndarray | list | None = None
    objects: list[dict] | None = None

    # The utility data
    data_len: int | None = None
    metadata: dict = field(default_factory=dict)
    gen_defaults: dict = field(default_factory=dict)

    def get_dict(self):
        data = {
            'coords': self.coords, 
            'features': self.features, 
            'annotation': self.annotation, 
            'label': self.label, 
            'objects': self.objects,
        }
        return {k: v for k, v in data.items() if v is not None}

    def assign_metadata(self, **kwargs):
        for key, value in kwargs.items():
            self.metadata[key] = value

    def filter(self, background_ratio):
        # We only pass the arrays/lists to filter_background_ratio
        filtered_data, data_len = filter_background_ratio(self.get_dict(), self.label, background_ratio)
        for k, v in filtered_data.items():
            setattr(self, k, v)
        self.data_len = data_len
    
    def fix_generator_args(self, batch_size=None, shuffle=None, iterate=None):
        if batch_size is not None: self.gen_defaults['batch_size'] = batch_size
        if shuffle is not None: self.gen_defaults['shuffle'] = shuffle
        if iterate is not None: self.gen_defaults['iterate'] = iterate
    
    def generator(self, batch_size=32, shuffle=True, iterate=True, propagate_loss=True):
        """
        Yields batches of data. If iterate=False, returns the entire 
        dataset with an added batch dimension.
        - propagate_loss is only when iterate=False
        """
        batch_size = self.gen_defaults.get('batch_size', batch_size)
        shuffle = self.gen_defaults.get('shuffle', shuffle)
        iterate = self.gen_defaults.get('iterate', iterate)

        # If not iterating, wrap all kwargs in an extra dimension and yield
        if not iterate:
            out = self.get_dict()
            out.update(self.metadata)
            out.update(propagate_loss=propagate_loss)
            yield out
            return

        # 1. Generate and shuffle indices
        data_len = self.data_len if self.data_len else len(self.coords)
        order = np.arange(data_len)
        if shuffle:
            np.random.shuffle(order)
        
        # 2. Iterate through the order in steps of batch_size
        full_dict = self.get_dict()
        for i in range(0, data_len, batch_size):
            batch_indices = order[i : i + batch_size]
            
            # 3. Create the batch dictionary
            batch_dict = {'split_id': batch_indices.astype(int)}
            
            for key, data_array in full_dict.items():
                # Slice the data based on shuffled indices
                batch_dict[key] = np.array([data_array[idx] for idx in batch_indices])

            batch_dict.update(self.metadata)
            batch_dict.update(propagate_loss=True)
            yield batch_dict


def aggr_patch_into_tile(embeddings, patch_to_tile):
    """Merge patch tokens into tile-level features via aggregation.
    
    Args:
        embeddings (dict): Dictionary containing class_token and optionally patch_tokens
    
    Returns:
        np.ndarray: Tile-level features, either (N, F) or (N, 2F) depending on strategy
    """

    # 1. Max or Mean the patch tokens
    if patch_to_tile in ["max", "mean"]:

        # Get pre-aggregated features
        if patch_to_tile == 'max':
            patch_features = embeddings.get("max_patch_token")
        elif patch_to_tile == 'mean':
            patch_features = embeddings.get("mean_patch_token")
            
        # If not available, aggregate from patch tokens
        if patch_features is None:
            if embeddings.get("patch_tokens") is None:
                return embeddings['class_token']
            else:
                patch_features = aggregate_patch_tokens(embeddings['patch_tokens'], patch_to_tile, is_torch=False)

        # Concatenate with class token
        aggr_features = np.concatenate([embeddings['class_token'], patch_features], axis=-1)  # (B, F+F) 
    
    # 2. Dont aggregate
    if patch_to_tile in ["discard", False]:
        aggr_features = embeddings['class_token']
    
    return aggr_features
    
def is_coord_a_group(coords_raw):
    """Check if coordinates contain a group_id column (3rd column).

    Args:
        coords_raw (np.ndarray): (N, 2) or (N, 3) coordinate array.

    Returns:
        bool: True if coordinates have 3 columns (x, y, group_id).
    """
    return bool(coords_raw.shape[1] == 3)

def ungroup_data_features(coords_raw, feats_raw, extra_data=None):
    """Split grouped data back into per-group lists.

    Separates arrays that were grouped by ``group_class_tokens`` back
    into individual groups based on the group_id in the 3rd coordinate
    column.

    Args:
        coords_raw (np.ndarray): (M, 3) coordinates [x, y, group_id].
        feats_raw (np.ndarray): (M, F) feature vectors.
        extra_data (np.ndarray, optional): (M, C) additional data
            (e.g. annotations) to split in parallel.

    Returns:
        tuple: (coords_list, feats_list, extra_data_list) where each
            is a list of arrays split by group_id. ``extra_data_list``
            is None if ``extra_data`` was not provided.
    """
    # 1. Sort by group_id to ensure groups are contiguous
    group_ids = coords_raw[:, 2]
    sort_idx = np.argsort(group_ids)

    coords_sorted = coords_raw[sort_idx]
    feats_sorted = feats_raw[sort_idx]
    
    # 2. Find the transition points where group_id changes
    sorted_groups = coords_sorted[:, 2]
    # np.diff identifies where the value changes; +1 to get the start of the next group
    change_indices = np.where(np.diff(sorted_groups) != 0)[0] + 1
    
    # 3. Split the arrays into lists of arrays
    # coords_list will contain arrays of shape (N1, 2), (N2, 2), etc.
    coords_split = np.split(coords_sorted[:, :2], change_indices)
    feats_list = np.split(feats_sorted, change_indices)
    
    extra_data_list = None
    if extra_data is not None:
        ann_sorted = extra_data[sort_idx]
        extra_data_list = np.split(ann_sorted, change_indices)
        
    return coords_split, feats_list, extra_data_list

def bag_data_features(coords_raw, feats_raw, ann_raw, objects):
    """Slice features into annotation-region bags for MIL training.

    Each bag corresponds to a spatial region defined by an annotation
    shape, containing only the tiles that overlap that region.

    Args:
        coords_raw (np.ndarray): (N, 2) tile coordinates.
        feats_raw (np.ndarray): (N, F) feature vectors.
        ann_raw (np.ndarray): (N, C) class fraction matrix.
        objects (list[dict]): Annotation bags from ``label_coords_by_json``,
            each with ``'label'`` (list) and ``'ids'`` (np.ndarray of indices).

    Returns:
        tuple: (coords_list, feats_list, ann_list, label_list) where each
            is a list with one entry per bag.
    """
    coords_list = []
    feats_list = []
    ann_list = []
    label_list = []

    for obj_info in objects:
        indices = obj_info['ids']
        if len(indices) == 0:
            continue
            
        # Slice raw arrays using indices directly
        coords_list.append(coords_raw[indices])
        feats_list.append(feats_raw[indices])
        
        if ann_raw is not None:
            ann_list.append(ann_raw[indices])
        
        # Create One-Hot Bag Label from the stored label_id
        label_list.append(obj_info['label'])

    return coords_list, feats_list, ann_list, label_list
    

def filter_background_ratio(pre_data, label_list, bg_ratio=0.20):
    """
    Keeps all foreground tiles and samples background tiles so they 
    make up exactly bg_ratio (e.g., 20%) of the final dataset.
    """
    labels = np.array(label_list)
    # 1. Identify Foreground vs Background
    # Foreground = anything that isn't 100% background (label[0] < 1.0)
    # Background = purely empty tiles (label[0] == 1.0)
    is_fg = labels[:, 0] < 1.0
    is_bg = ~is_fg
    
    fg_indices = np.where(is_fg)[0]
    bg_indices = np.where(is_bg)[0]
    
    n_fg = len(fg_indices)
    
    # 2. Calculate budget for Background
    # Math: n_bg_allowed / (n_fg + n_bg_allowed) = bg_ratio
    # Solve for n_bg_allowed: n_bg_allowed = (bg_ratio * n_fg) / (1 - bg_ratio)
    n_bg_allowed = int((bg_ratio * n_fg) / (1 - bg_ratio))
    
    # Safety check: If we have fewer background tiles than allowed, just keep them all
    n_bg_to_sample = min(len(bg_indices), n_bg_allowed)
    
    # 3. Randomly sample from the background pool
    sampled_bg_indices = np.random.choice(bg_indices, size=n_bg_to_sample, replace=False)
    
    # 4. Combine and sort to maintain some level of spatial order
    final_indices = np.concatenate([fg_indices, sampled_bg_indices])
    final_indices.sort()
    
    # 5. Filter the pre_data dictionary
    filtered_data = {
        k: v[final_indices] if isinstance(v, np.ndarray) else [v[i] for i in final_indices]
        for k, v in pre_data.items() if v is not None
    }
    
    # print(f"Foreground: {n_fg}, Background kept: {n_bg_to_sample} (Target Ratio: {bg_ratio*100}%)")
    return filtered_data, len(final_indices)