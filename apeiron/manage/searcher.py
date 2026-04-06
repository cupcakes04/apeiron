import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import faiss
from typing import Literal
import pandas as pd
from apeiron.utils import convert_to_list

class Searcher:
    """
    Transforms (N, F) variable-length local descriptors into a single 
    fixed-length (K*F) global descriptor using FAISS GPU KMeans.
    """
    def __init__(self, num_clusters=16, use_gpu=True, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.use_gpu = use_gpu
        self.centers = None
        self.global_index: faiss.IndexFlatL2 # Fast FAISS index for assigning tiles to centers
        self.search_index: faiss.IndexFlatL2
        self.reset_searcher()

    def reset_searcher(self):
        self.vectors_features = []
        self.vectors_coords = []
        self.vectors_id = []
        self.vectors_image = []
        self.vector_indexes = {}
        
        
    # |-----------------------------------------------|
    # |---------------------- VLAD -------------------|
    # |-----------------------------------------------|

    # Global Descriptor: VLAD (Vector of Locally Aggregated Descriptors)

    def fit(self, all_descriptors: np.ndarray):
        """
        Fits the codebook on a large representative sample of local descriptors.
        all_descriptors: (total_N, F)
        """
        # FAISS strictly requires C-contiguous float32 arrays
        all_descriptors = np.ascontiguousarray(all_descriptors, dtype=np.float32)
        d = all_descriptors.shape[1]
        
        # 1. Train KMeans on GPU (insanely faster than sklearn)
        print(f"Training FAISS KMeans with K={self.num_clusters} on GPU={self.use_gpu}...")
        self.kmeans = faiss.Kmeans(d=d, k=self.num_clusters, niter=20, verbose=False, gpu=self.use_gpu)
        self.kmeans.train(all_descriptors)
        self.centers = self.kmeans.centroids
        
        # 2. Build a FAISS index from the centers so we can quickly assign new tiles
        # We keep this assignment index on CPU because for K=16 to K=256, 
        # CPU is practically instantaneous, and it avoids CUBLAS errors on small matrices.
        self.global_index = faiss.IndexFlatL2(d)
        self.global_index.add(np.ascontiguousarray(self.centers, dtype=np.float32))

    def fit_from_generator(self, descriptor_generator, max_samples=500000):
        """
        Fits the codebook by dynamically sampling tiles from a generator.
        This prevents out-of-memory errors when processing 1000s of slides.
        At peak, it holds `max_samples * 2` tiles in memory.
        E.g., 500,000 tiles of 1536-dim float32 takes ~3GB of RAM.
        """
        buffer = []
        current_size = 0
        
        for desc in descriptor_generator:
            buffer.append(desc['features'])
            current_size += desc['features'].shape[0]
            
            # If buffer gets too large, merge, shuffle, and downsample to max_samples
            if current_size > max_samples * 2:
                merged = np.vstack(buffer)
                indices = np.random.choice(merged.shape[0], max_samples, replace=False)
                buffer = [merged[indices]]
                current_size = max_samples
                
        # Final merge and truncate before fitting
        if buffer:
            merged = np.vstack(buffer)
            if merged.shape[0] > max_samples:
                indices = np.random.choice(merged.shape[0], max_samples, replace=False)
                merged = merged[indices]
            self.fit(merged)
        else:
            raise ValueError("Generator yielded no data.")

    def compute(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Computes the VLAD vector for a single slide.
        descriptors: (N, F) features for a single slide
        Returns: (K * F) 1D numpy array
        """
        assert self.centers is not None, "Must fit codebook first!"
        
        # Ensure contiguous float32 for FAISS
        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        N, F = descriptors.shape
        K = self.num_clusters
        
        # 1. Vectorized GPU Assignment (Which cluster is each tile closest to?)
        # Returns distances and indices of the closest cluster for each tile
        _, labels = self.global_index.search(descriptors, 1) 
        labels = labels.flatten()
        
        # 2. Vectorized Residual Accumulation
        vlad_vector = np.zeros((K, F), dtype=np.float32)
        for i in range(K):
            # We still use a loop for K (small), but the math inside is vectorized
            mask = (labels == i)
            if np.any(mask):
                # Intra-normalization happens here (optional but recommended)
                diff = descriptors[mask] - self.centers[i]
                vlad_vector[i] = np.sum(diff, axis=0)
                
                # --- Best Practice: Intra-normalization ---
                norm = np.linalg.norm(vlad_vector[i]) + 1e-8
                vlad_vector[i] /= norm 

        # 3. Flatten and Global Normalization
        vlad_vector = vlad_vector.flatten()
        
        # Power normalization (SSR)
        vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))
        
        # Final L2
        vlad_vector /= (np.linalg.norm(vlad_vector) + 1e-8)
        return vlad_vector

    def build_index(self, descriptor_generator, compute=True):
        """
        Yields VLAD vectors one by one for a stream of slides.
        descriptor_generator: Yields numpy arrays of shape (N_i, F)
        Returns: A generator that yields (K * F) 1D numpy arrays
        """
        self.reset_searcher()

        # Extract and compute the global vectors
        for desc in descriptor_generator:
            self.vectors_id.extend(convert_to_list(desc['id']))
            features = self.compute(desc['features']) if compute else desc['features']
            self.vectors_features.append(features)
            self.vectors_coords.append(desc['coords'])
            self.vectors_image.append(desc['img_emb'])

        self.vector_indexes = {vid: i for i, vid in enumerate(self.vectors_id)}
        self.vectors_features = np.vstack(self.vectors_features)
        self.vectors_coords = np.vstack(self.vectors_coords)
        self.vectors_image = np.vstack(self.vectors_image)
        
        # Built index
        self.vector_dim = self.vectors_features.shape[1]
        self.search_index = faiss.IndexFlatL2(self.vector_dim)
        self.search_index.add(np.ascontiguousarray(self.vectors_features, dtype=np.float32))


    # |-----------------------------------------------|
    # |------- Indexing & Retrieval: FAISS -----------|
    # |-----------------------------------------------|
    
    # Manages FAISS indexing and fast searching of global slide descriptors


    def prepare_vector(self, vec: np.ndarray|str|list[str], mode: Literal['feat', 'img']):
        # 1. Handle ID lookup (Single string or List of strings)
        if isinstance(vec, (str, list, tuple)):
            
            # Normalize to a list so we can iterate
            search_ids = [vec] if isinstance(vec, str) else vec
            indices = [self.vector_indexes[vid] for vid in search_ids]

            # Pull the correct rows from your feature/image storage
            if mode == 'feat':
                vec = self.vectors_features[indices]
            elif mode == 'img':
                vec = self.vectors_image[indices]

        # 2. Final Check & Shape Normalization
        if isinstance(vec, np.ndarray):
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
        
        return vec


    def find_similar_text(self, img_emb: np.ndarray|str|list[str], wrd_emb: np.ndarray, mode: Literal['wrd', 'img']):
        """
        Calculate scores
        - `wrd`
            - matching score for each wrd: (4, H) @ (H, 3) = (4, 3)
        - `img`
            - matching score for each img: (10, H) @ (H, 2) = (10, 2)
        # """

        img_emb = self.prepare_vector(img_emb, mode='img')
        if mode == 'wrd':
            row_labels = [f"wrd_{i}" for i in range(wrd_emb.shape[0])] 
            col_labels = [f"img_{i}" for i in range(img_emb.shape[0])]
            scores = wrd_emb @ img_emb.T
        if mode == 'img':
            row_labels = [f"img_{i}" for i in range(img_emb.shape[0])] 
            col_labels = [f"wrd_{i}" for i in range(wrd_emb.shape[0])]
            scores = img_emb @ wrd_emb.T

        # Create the DataFrame
        return pd.DataFrame(scores, index=row_labels, columns=col_labels)


    def find_similar_feat(self, query_feat_vec: np.ndarray|str, top_k: int = 5):
        """
        query_features: (1, D) or (D,)
        """
        feat_vec = self.prepare_vector(query_feat_vec, mode='feat')
            
        distances, indices = self.search_index.search(np.ascontiguousarray(feat_vec, dtype=np.float32), top_k)
        
        # Efficiently build a list of row data
        results_data = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results_data.append({
                'id': self.vectors_id[idx], 
                'distance': dist
            })
        return pd.DataFrame(results_data)

    # The Spatial Step: Finding Regions of Interest (Region Score)
    @staticmethod
    def find_similar_regions(
        target_features: np.ndarray,
        target_coords: np.ndarray, 
        query_features: np.ndarray, 
        smoothing_radius: float = 500.0,
        similarity_threshold: float = 0.8,
        min_tiles: int = 100,
        eps_dbscan: float = 1000.0,
        min_samples: int = 3
    ):
        """
        Given a set of query features (e.g. multiple tiles from a region of interest), 
        find and cluster spatially contiguous regions in a target slide that match.
        
        target_features: (N, F) features of the target slide
        target_coords: (N, 2) physical coordinates (x, y) of the tiles
        query_features: (Q, F) or (F,) the set of feature vectors you are searching for
        """
        
        # Ensure query_features is 2D: (Q, F)
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
            
        # --- 1. Compute Base Similarities (Cosine) ---
        # Normalize both target and query features
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute dot product between all targets and all queries -> (N, Q)
        similarity_matrix = np.dot(target_norm, query_norm.T)
        
        # Best Practice: A target tile is as relevant as its highest match to ANY of the query tiles.
        # So we collapse the Q dimension by taking the max. -> (N,)
        raw_scores = np.max(similarity_matrix, axis=1)
        
        # --- 2. Spatial Smoothing (Heatmap generation) ---
        # We use a KDTree to quickly find all tiles within `smoothing_radius`
        tree = cKDTree(target_coords)
        smoothed_scores = np.zeros_like(raw_scores)
        
        for i, coord in enumerate(target_coords):
            # Find indices of tiles within spatial radius
            neighbors = tree.query_ball_point(coord, r=smoothing_radius)
            # Region Score is the average similarity of all neighboring tiles
            smoothed_scores[i] = np.mean(raw_scores[neighbors])
            
        # --- 3. Filter Highly Similar tiles ---
        high_score_mask = smoothed_scores > similarity_threshold
        
        if min_tiles is not None and np.sum(high_score_mask) < min_tiles:
            # Dynamically lower threshold to guarantee at least min_tiles
            take_n = min(min_tiles, len(smoothed_scores))
            # Get indices of the top 'take_n' scores
            top_indices = np.argsort(smoothed_scores)[-take_n:]
            high_score_coords = target_coords[top_indices]
            high_score_vals = smoothed_scores[top_indices]
        else:
            # Hard filter using the similarity threshold
            high_score_coords = target_coords[high_score_mask]
            high_score_vals = smoothed_scores[high_score_mask]
        
        if len(high_score_coords) == 0:
            return [] # No regions passed the threshold
            
        # --- 4. Density-Based Clustering (DBSCAN) ---
        # Group tiles that are close to each other geographically
        clustering = DBSCAN(eps=eps_dbscan, min_samples=min_samples).fit(high_score_coords)
        labels = clustering.labels_
        
        # --- 5. Ranking the Groups (Composite Score) ---
        regions = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue # -1 designates noise/outliers in DBSCAN
                
            cluster_mask = (labels == label)
            cluster_coords = high_score_coords[cluster_mask]
            cluster_scores = high_score_vals[cluster_mask]
            
            # Metric 1: Average Similarity
            avg_sim = np.mean(cluster_scores)
            # Metric 2: Cluster Mass (Size of region)
            cluster_mass = len(cluster_scores)
            
            # Composite score weighting size and similarity
            composite_score = avg_sim * np.log1p(cluster_mass)
            
            regions.append({
                'cluster_id': label,
                'composite_score': composite_score,
                'avg_similarity': float(avg_sim),
                'tile_count': int(cluster_mass),
                'centroid': np.mean(cluster_coords, axis=0).tolist(),
                'tile_coords': cluster_coords.tolist()
            })
            
        # Sort regions heavily favoring the highest composite score
        regions.sort(key=lambda x: x['composite_score'], reverse=True)
        return pd.DataFrame(regions)