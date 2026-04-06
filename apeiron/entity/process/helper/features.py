import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import torch
from typing import Literal
from sklearn.decomposition import PCA
from dataclasses import dataclass, field

@dataclass
class ProcessedExt:
    # Basic Extractions
    coords: np.ndarray | None = None    # (N, 2)
    features: np.ndarray | None = None   # (N, F)
    feats_size: int | None = None
    coords_size: int | None = None
    objects: list[dict] | None = None

    # Feature manipulations
    feats_color: np.ndarray  | None = None    # (N, 3)
    feats_score: np.ndarray | None = None     # (N, 1)
    feats_clusters: np.ndarray | None = None  # (N,)
    feats_color_map: np.ndarray | None = None # (n_clusters, 3) RGB color for each cluster

    def reset(self):
        """Clear all feature-related data to free memory."""
        self.coords = None
        self.features = None
        self.feats_size = None
        self.coords_size = None
        
    def assign(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_feat_color_data(self):
        return {
            'feats_color': self.feats_color, 'feats_size': self.feats_size, 
            'coords': self.coords, 'coords_size': self.coords_size, 
        }


def reduce_features(x: np.ndarray, dims: int = 3, nns: int = 15, md: float = 0.1, 
                    method: Literal["umap", "pca"] = 'pca') -> np.ndarray:
    """Reduce high-dimensional features to lower dimensions for visualization.
    
    Applies UMAP or PCA dimensionality reduction to transform features into
    a lower-dimensional space (typically 3D for RGB visualization). Results
    are normalized to [0, 1] range.
    
    Args:
        x (np.ndarray): (N, F) high-dimensional feature matrix
        dims (int): Target number of dimensions. Default 3 (for RGB)
        nns (int): Number of neighbors for UMAP. Controls local vs global structure.
            Lower values preserve local structure, higher values preserve global. Default 15
        md (float): Minimum distance for UMAP in [0, 1]. Controls cluster separation.
            Values near 0 create tight clusters, near 1 creates dispersed points. Default 0.1
        method (str): Reduction method - 'umap' or 'pca'. Default 'umap'
    
    Returns:
        np.ndarray: (N, dims) reduced and normalized features in [0, 1] range
    """
    if dims is None:
        return x
    
    # If already reduced to dims, return normalized copy
    if x.shape[1] == dims:
        reduced = x.copy()
        reduced -= reduced.min(axis=0)
        reduced /= reduced.max(axis=0)
        return reduced
    
    # Umap or PCA
    if method == 'umap':
        reducer = umap.UMAP(
            n_neighbors=nns,
            min_dist=md,
            n_components=dims,
            metric="manhattan",
            spread=0.5,
            random_state=2,
        )
        reduced = reducer.fit_transform(x)
    elif method == 'pca':
        pca = PCA(n_components=dims)
        reduced = pca.fit_transform(x)
    
    reduced -= reduced.min(axis=0)
    reduced /= reduced.max(axis=0)
    return reduced

def cluster_feats(feats, n_clusters):
    """Cluster features using K-means and assign colors to each cluster.
    
    Args:
        feats (np.ndarray): (N, 3) RGB features to cluster
        n_clusters (int): Number of clusters to create
    
    Returns:
        tuple:
            - feats_cluster (np.ndarray): (N,) cluster indices (0 to n_clusters-1)
            - color_map (np.ndarray): (n_clusters, 3) RGB color for each cluster
    """
    # 1. Perform Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    feats_cluster = kmeans.fit_predict(feats) # Returns 1D array (N,)
    
    # 2. Generate the Color Map (Lookup Table)
    cmap = plt.get_cmap('tab20', n_clusters)
    # Generate all colors at once: (n_clusters, 3)
    color_map = cmap(np.arange(n_clusters))[:, :3] 
    
    return feats_cluster, color_map

def rank_feats(feats, query_indices, scoring_mode='rank', sigma=None):
    """
    Rank patches by similarity to query region using specified scoring mode.

    Args:
        feats (np.ndarray): (N, F) feature matrix.
        query_indices (int or list[int]): Query patch index/indices.
        scoring_mode (str): Scoring mode: `rank`, `cosine`, or `gaussian`.
        sigma (float or None): Bandwidth (0.1~10) for Gaussian (used only if mode='gaussian').
            * None or 'auto' uses standard deviation to estimate.
            
            Example effects of sigma on score decay:
                Small Sigma = 5
                    Distance:   0    1     2     3     4     5
                    Score:     1.0  0.1  0.01  ~0   ~0    ~0

                Large Sigma = 30
                    Distance:   0    1     2     3     4     5
                    Score:     1.0  0.85  0.6   0.3   0.1   0.01
    Returns:
        np.ndarray: (N, 1) score array in [0, 1].
    """
    if isinstance(query_indices, int):
        query_indices = [query_indices]

    region_feats = feats[query_indices]
    centroid = region_feats.mean(axis=0, keepdims=True)

    if scoring_mode == 'rank':
        dists = np.linalg.norm(feats - centroid, axis=1)
        max_d = dists.max()
        min_d = dists.min()
        scores = 1 - ((dists - min_d) / (max_d - min_d + 1e-8))

    elif scoring_mode == 'cosine':
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        centroid_norm = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)
        scores = feats_norm @ centroid_norm.T  # shape (N, 1)
        scores = (scores + 1) / 2  # map from [-1,1] to [0,1]

    elif scoring_mode == 'gaussian':
        dists = np.linalg.norm(feats - centroid, axis=1)
        if sigma in [None, 'auto']:
            sigma = np.std(dists)
        scores = np.exp(- (dists**2) / (2 * sigma**2))

    else:
        raise ValueError("scoring_mode must be one of: 'rank', 'cosine', 'gaussian'")

    scores = scores.reshape(-1, 1)
    
    return scores

def feats_score_to_coolwarm(feats_score):
    """Convert similarity scores to RGB colors using coolwarm colormap.
    
    Maps low scores to blue (cool) and high scores to red (warm) for
    intuitive visualization of similarity to query region.
    
    Args:
        feats_score (np.ndarray): (N, 1) or (N,) similarity scores
    
    Returns:
        np.ndarray: (N, 3) RGB colors in [0, 1] range
    """
    # 1. Ensure it is a 1D array for processing
    scores = feats_score.flatten()
    
    # 2. Apply the 'coolwarm' colormap
    cmap = plt.get_cmap('coolwarm')
    feats_score_colored = cmap(scores)[:, :3]  # Strip alpha channel, keep RGB
    return feats_score_colored

def ann_percentages_to_rgb(ann_percentages, class_colors=None):
    """
    Converts (N, C) class fractions into (N, 3) RGB colors.
    
    Args:
        ann_percentages: (N, C) float array.
        class_colors: (C, 3) array of RGB values (0-1). 
                      If None, generates random colors.
    Returns:
        rgb_ann: (N, 3) uint8 array.
    """
    N, C = ann_percentages.shape

    # 1. Setup Colors
    if class_colors is None:
        # Generate random colors for all classes
        cmap = plt.get_cmap()
        color_map = np.array([cmap(i)[:3] for i in range(C)], dtype=np.float32)
    else:
        color_map = np.array(class_colors, dtype=np.uint8)
    
    # 2. FORCE Class 0 to be Black
    color_map[0] = [0, 0, 0]

    # 3. Create One-Hot Map (C, C)
    # This is an identity matrix where row i is the one-hot vector for class i
    one_hot_map = np.eye(C, dtype=np.float32)

    # 4. Find dominant class
    max_indices = np.argmax(ann_percentages, axis=1)
    
    # 5. Create the RGB map
    rgb_ann = color_map[max_indices]

    # 6. Safety Check: If a tile is 0.0 across ALL classes, 
    # it must be black regardless of what argmax says.
    # (argmax returns 0 if all values are equal/zero)
    no_signal = np.max(ann_percentages, axis=1) == 0
    rgb_ann[no_signal] = [0, 0, 0]

    # 7. Package Metadata
    color_map = {
        'color': color_map,     # (C, 3)
        'class': one_hot_map,   # (C, C)
    }
    return rgb_ann, color_map
        
def obj_to_NC(obj_list):
    all_labels = []

    for obj in obj_list:
        n_ids = len(obj['ids'])
        labels = obj['labels']
        
        # Repeat the label row N times
        repeated_labels = np.tile(labels, (n_ids, 1))
        all_labels.append(repeated_labels)

    # Stack all blocks vertically
    return np.vstack(all_labels)

def obj_to_rgb(obj_list, coords):

    num_objs = len(obj_list)
    
    # Use base coordinates
    N = coords.shape[0]

    rgb_obj = np.zeros((num_objs, N, 3), dtype=np.float32)

    # Generate distinct colors for objects
    np.random.seed(42)
    obj_colors = np.random.uniform(0, 1, size=(num_objs, 3)).astype(np.float32)

    # Create object color_map metadata mimicking ann_percentages_to_rgb structure
    C = len(obj_list[0]['labels'])
    color_map = {
        'color': (obj_colors).astype(np.float32), # (O, 3)
        'class': np.array([obj['labels'] for obj in obj_list]), # (O, C)
    }

    for o_idx, obj in enumerate(obj_list):
        ids = obj['ids']
        if len(ids) > 0:
            rgb_obj[o_idx, ids] = obj_colors[o_idx]
    
    return rgb_obj, color_map



def normalise(ann, mode: Literal['minmax', 'rank', 'topk'] = 'topk') -> np.ndarray:
    if (ann.ndim == 1):
        ann = ann[:, np.newaxis]
        
    # Input shape: (N, C)

    if mode == 'minmax':
        # We calculate min/max across dim 0 (the N tiles)
        ann_min = np.min(ann, axis=0, keepdims=True)
        ann_max = np.max(ann, axis=0, keepdims=True)

        # Standard formula
        return (ann - ann_min) / (ann_max - ann_min + 1e-8)

    elif mode == 'rank':
        # 1. Get the rank of each element along the N dimension
        # .argsort().argsort() is the standard trick to get ranks in PyTorch
        ranks = ann.argsort(axis=0).argsort(axis=0).astype(float)

        # 2. Scale ranks to [0, 1] range
        # (Rank / (N - 1))
        return ranks / (ann.shape[0] - 1)
        
    elif mode == 'topk':
        k = max(1, int(ann.shape[0] * 0.1)) # Ensure k is at least 1
        
        # Get values and indices
        topk_values, _ = torch.topk(ann, k, dim=0)
        kth_values = topk_values[-1, :] # (C,)
        
        # NEW: Add a zero-floor. 
        # This ensures that even if it's in the top 10%, 
        # if the logit is negative (background), we don't treat it as a "hot" tile.
        threshold = torch.clamp(kth_values, min=0.0) 
        
        mask = ann >= threshold
        return ann * mask.float()

def get_grid_coords(coords):
    if coords.shape[1] <= 2:
        return coords

    # 1. Sort by Grid ID (3rd column)
    sort_idx = np.argsort(coords[:, 2])
    sorted_coords = coords[sort_idx]

    # 2. Find the starting index of each unique Grid ID
    # return_index gives the position of the FIRST occurrence of each ID
    _, starts = np.unique(sorted_coords[:, 2], return_index=True)

    # 3. Just take the x, y from those starting positions
    # This assumes the first point in the data is the top-left
    return sorted_coords[starts, :2]