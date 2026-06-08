# Apeiron Entity (`apeiron/entity`)

## Overview
The `entity` directory acts as the data-wrangling engine and IO interface of Apeiron. It bridges the gap between raw physical assets (Whole Slide Images, TIFF/JSON annotations, external `.pt` features) and the strict tensors expected by downstream AI models.

It is split into three main directories:
1.  **`process`**: Standardizes arrays, formats dimensional representations, and packages items into iterators.
2.  **`reader`**: Interfaces with `.svs`/`.tiff` files via OpenSlide to extract spatial metadata and pixels.
3.  **`dataset`**: Wraps the processed data and readers into standard `torch.utils.data.Dataset` objects for model ingestion.

---

## 1. Process (`apeiron/entity/process`)

The `process` module handles standardising varying patch and token shapes, projecting hierarchical coordinates, generating color-mapped dimensional reductions, and constructing strictly formatted batch generators.

### Dataclasses
These dataclasses are commonly used externally (e.g., in `Collector`, `Inferencer`, and Notebooks) to maintain a strict typing contract throughout the pipeline.

**1. `PreData` (`helper/preprocess.py`)**
Encapsulates data ready to be streamed to a model. The `generator()` method allows automatic batching, shuffling, and gradient accumulation logic (via `propagate_loss`).
```python
@dataclass
class PreData:
    coords: np.ndarray | list       # Tile coordinates, shape (N, 2)
    features: np.ndarray | list     # Feature embeddings, shape (N, F)
    annotation: np.ndarray | list | None   # Target annotation arrays, shape (N, C)
    label: np.ndarray | list | None        # Slide-level label, shape (C,) or int
    objects: list[dict] | None      # ROI-level annotations, list of Dict, keys: {'label' -> list (length C), 'ids' -> np.ndarray (shape N_roi,)}
    
    # Utility data
    data_len: int | None
    metadata: dict                  # Slide ID, model modes, metadata
    gen_defaults: dict              # Default kwargs for generator
```

**2. `ProcessedExt` (`helper/features.py`)**
Used to store fully extracted features and their computed clustering/dimensionality reduction metrics for the current slide being analyzed.
```python
@dataclass
class ProcessedExt:
    # Basic Extractions
    coords: np.ndarray | None       # Base coordinates (N, 2)
    features: np.ndarray | None     # Primary feature embeddings (N, F)
    coords_size: int | None
    feats_size: int | None
    objects: list[dict] | None      # Extracted ROIs mapped to the features
    
    # Feature manipulations
    feats_color: np.ndarray | None     # (N, 3) RGB values for UMAP/PCA plots
    feats_score: np.ndarray | None     # (N, 1) Anomaly/Similarity ranking scores
    feats_clusters: np.ndarray | None  # (N,) Int cluster assignments (e.g. KMeans)
    feats_color_map: np.ndarray | None # (n_clusters, 3) RGB color mapping for each cluster
```

**3. `VizData` (`visualise.py`)**
Defines how coordinates and matrices should map onto the visual thumbnails.
```python
@dataclass
class VizData:
    draw_coords: np.ndarray              # Where to draw
    draw_colors: np.ndarray
    color_map: tuple | list | None = None
    show: Callable | None = None
    hist: Callable | None = None
```

### Modalities and Formats
The processing pipeline strictly adheres to specific tensor/array shapes representing multiple modalities:

*   **`features` `(B, N, F)` or `(N, F)`:** The core visual representation. Can be pooled tile features, hierarchical patch tokens, or globally aggregated representations.
*   **`coords` `(N, 2)`:** Physical spatial mappings `(X, Y)` mapped to the top-level slide magnification.
*   **`label` `(C,)`:** Slide-level target labels (e.g., survival status, molecular subtype).
*   **`annotation` `(N, C)`:** Tile-level predictions or ground truth. Each element `(i, c)` represents the fraction/probability of class `c` present in tile `i`.
*   **`objects` `list[dict]`:** Used for dynamic Region of Interest (ROI) MIL or DETR architectures. Each dictionary in the list represents a bounding bag of tiles with keys `label` (a list of length `C` representing the ROI's class fraction/one-hot) and `ids` (a 1D `np.ndarray` of shape `(N_roi,)` containing the exact tile indices from `N` that belong to this ROI).
*   **`text` `str` or `list[str]`:** Prompt context for Visual-Language Models or generative decoders.

### Configuration Parameters
When interacting with `Processor` or `Annotator`, you can specify several arguments that drastically change how annotations and data are batched and formatted.

*   **`ann_type`**: Determines how `ann_path` is loaded.
    *   `'shape'`: Reads from GeoJSON polygons (e.g. QuPath exports). Calculates intersection coverage of tiles vs polygons.
    *   `'pixel'`: Reads from a single-channel TIFF mask. Extracts pixel values to generate class fractions.
*   **`supervision` (bool)**: If `True`, changes the model from a Slide-level MIL into an ROI-level model. Instead of yielding one massive bag of `(N, F)` features per slide, it extracts specific annotated regions and groups them into `objects` dictionaries, setting `batch_size = 1` for heterogeneous ROI bagging.
*   **`background_ratio` (float)**: e.g., `0.20`. Used during tile-level extraction (only in window_level=grid, grouped coords). It keeps all foreground tiles (where annotation > 0) and randomly subsamples background tiles so that the background makes up exactly `background_ratio` (20%) of the final generated dataset.

### Expected Raw Annotation Formats
When bringing your own annotations to Apeiron, they must strictly follow these structural requirements based on the `ann_type` used:

**1. Shape Format (`ann_type = 'shape'`)**
Requires a `.json` file containing a dictionary of annotation objects. Each object represents an ROI or polygon and must contain `type`, `properties`, and `geometry`.
```json
{
  "region_1": {
    "type": "polygon",
    "properties": {
      // Can be a single integer class index:
      "label_id": 1
      // OR a multi-class probability dictionary:
      // "label_id": {"ids": [1, 2], "weights": [0.7, 0.3]}
    },
    "geometry": {
      // Polygon points mapped to the base magnification coordinate space
      "vertices": [[x1, y1], [x2, y2], [x3, y3], ...] 
    }
  },
  "region_2": {
    "type": "ellipse",
    "properties": { "label_id": 2 },
    "geometry": { "center": [cx, cy], ... }
  }
}
```

**2. Pixel Format (`ann_type = 'pixel'`)**
Requires a `.tiff` or `.tif` file. It expects an image mask that maps directly over the slide space (usually downscaled). 
* Ideally, the arrays are `(H, W, C)` dimensional tensors where each channel `C` contains the fractional probabilities (0.0 to 1.0) or binary occurrences of that specific class across the pixels.
* **Automatic Internal Processing**: 
  * If you provide a raw **single-channel `(H, W)` mask** containing integer class indices, Apeiron will automatically one-hot encode it into the required `(H, W, C)` binary mask using the internal `mask_to_binary()` helper.
  * If the mask's resolution does not perfectly align with the expected coordinate downscale factor, Apeiron will automatically interpolate and scale it using `resize_mask()` to ensure perfect spatial alignment with the tiles. (into 16x downsample, optimal factor btw, so downscale them first too)

**3. Direct Dictionary Format**
Instead of loading files from disk, you can directly pass a pre-loaded dictionary to `process_annotations` as the `ann_path` argument containing pre-computed targets. It can contain either or both of these keys:
```python
{
  "annotation": np.ndarray,      # (N, C) class fractions/activations matrix
  "objects": list[dict]          # ROI-level annotations list (as described above)
}
```

### Core Modules (`process/`)
*   **`Processor` (`processor.py`)**: A heavy-weight class that uses Multiple Inheritance (`Reader`, `Annotator`, `Visualiser`) to expose a unified API for a slide.
    *   `assign_features()`: Handles the injection of `embeddings`. It can flatten hierarchies (merging class tokens and patch tokens via `max`, `mean`, or `discard`) down to flat `(N, F)` tensors.
    *   `slide_preprocessor()` & `tile_preprocessor()`: Packages the internal variables, coordinates, and annotations into a `PreData` object.
    *   `postprocessor()`: Ingests raw outputs (`mdata`) from Inferencers and integrates them back into the internal state (e.g., appending model attention weights back to visualizable space).
*   **`Annotator` (`annotate.py`)**: Handles intersection algorithms and spatial geometry.
    *   `process_annotations(coords, tile_size, ann_path, active_coords=False)`: Standard entry point to load and parse annotations. Safely dispatches based on file extensions or accepts a pre-loaded annotations dictionary (as described in the Direct Dictionary Format above) or `None` (resets annotations).
    *   `label_coords_by_json()`: Intersects GeoJSON boundary polygons with strict `(N, 2)` coordinate grids to determine which tiles belong to which annotation class.
    *   `mask_to_binary()` & `resize_mask()`: Resolves raw bitmap masks into clean, scalable `(N, C)` matrices.
*   **`Visualiser` (`visualise.py`)**: Generates high-quality NumPy array visualizations and Matplotlib plots.
    *   `draw_on_thumbnail()`: Core canvas loop mapping `VizData` structs dynamically over a downsampled WSI. 
    *   `create_feature_viz()`: Translates feature embeddings into RGB colors (PCA/UMAP) or scalar scores.
    *   `create_annotation_viz()`: Computes an RGB overlay layer using class colors against the `(N, C)` annotation matrix.

---

## 2. Reader (`apeiron/entity/reader`)

Handles direct interactions with Whole Slide Images (WSI), thumbnail extraction, coordinate generation, and mask tissue detection.

### Core Modules (`reader/`)
*   **`Reader` (`reader.py`)**: Inherits from `Thumbnailer` and manages the main Slide handle context.
    *   `setup_slide()`, `setup_tiles()`: Prepares the internal scaling and metadata to ensure 1-to-1 parity between micron-per-pixel (mpp) readings.
    *   `read_window()`: Dynamically extracts raw RGB `Pillow` images or NumPy arrays based on precise spatial coordinates.
    *   `create_tile_coords()`, `create_windowed_tile_coords()`: Generates a dense grid of `(N, 2)` spatial coordinates spanning the whole slide, or hierarchically windowed coordinates.
*   **`Thumbnailer` (`thumbnail.py`)**: Responsible for extracting visual overviews.
    *   `get_slide_thumbnail()`, `get_masked_thumbnail()`: Extracts ultra-low-resolution images for plotting overlays without blowing up memory.
    *   `normalise_image()`: Exposes `macenko` stain normalization logic using built-in matrix decomposers.
*   **`mask.py`**:
    *   `full_mask_to_tile_coords()` & `mask_to_tile_coords()`: Computes Otsu-thresholded tissue detection masks and morphological closings on thumbnails, mapping the boolean tissue fields precisely back to native WSI spatial grids to discard empty background.

---

## 3. Dataset (`apeiron/entity/dataset`)

Contains native PyTorch `Dataset` implementations that tightly wrap the generators and WSI loaders into standard iterators ready for `DataLoader` ingestion during distributed or multiprocessed training.

### Core Modules (`dataset/`)
*   **`SlideTiles` (`slide_tiles.py`)**: Maps `(N, 2)` integer coordinates against a single unified `slide_path`. Opens the slide handle inside the `__getitem__` logic to safely crop individual patches on-the-fly and apply TorchVision augmentations.
*   **`StandaloneTiles` (`standalone_tiles.py`)**: A classical directory-style image loader for standalone `.jpg` / `.png` crops (when patches have already been physically extracted to disk and lack a parent WSI context).
*   **`WindowTiles` (`window_tiles.py`)**: Similar to `StandaloneTiles` but respects spatial hierarchies, allowing the model to load macro-windows along with their constituent sub-patches simultaneously.
