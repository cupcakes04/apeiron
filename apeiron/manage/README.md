# Apeiron Manage API Reference (`apeiron/manage`)

## Overview
This document provides a highly detailed API reference for the exposed functions and data structures offered by the `manage` module. 

Since `Collector` inherits from `Manager` and `Searcher`, and acts as the primary `Operator` frontend (as seen in `operate.py`), this documentation is structured around the practical workflows and methods you will call when interacting with an `operator` or `manager` instance.

---

## 1. Project Configuration & Querying

Before running generations or training, the manager needs to know which project workspace to use and which subset of data to operate on.

### `setup_configs`
**Description**: Bootstraps the manager into a specific project workspace. Loads the `config.yaml`, `slide_dataset.csv`, and `tile_dataset.csv` into Pandas DataFrames and dictionaries.
*   **Inputs**:
    *   `project_path` *(str)*: Absolute or relative path to the project directory containing the config and dataset CSV files.
*   **Outputs**: 
    *   Mutates internal state: `self.config`, `self.slide_dataset`, `self.tile_dataset`.
    *   Initializes `self.analyzer = Analyzer(...)` with backbone configs.

### `query_slide_dataset` / `query_tile_dataset`
**Description**: Filters the loaded `slide_dataset` or `tile_dataset` to a specific subset of IDs for subsequent artifact generation or training.
*   **Inputs**:
    *   `slide_ids` / `tile_ids` *(list[str], optional)*: A list of specific UUIDs to filter down to.
    *   `select_all` *(bool, default=True)*: If `True`, selects the entire loaded dataset and ignores the IDs list.
*   **Outputs**:
    *   Returns the filtered `pd.DataFrame`.
    *   Mutates internal state: `self.selected_slide_dataset` or `self.selected_tile_dataset`.

---

## 2. Artifact Generation

These functions handle the heavy computations (ViT feature extraction, masking, thumbnail generation). They run in multiprocessing pools and safely cache results to disk via `ArtifactIO`.

### `generate_embeddings_slide` / `generate_embeddings_tile`
**Description**: Spawns multiple workers to read images, extract patch features using the `Backbone`, apply the specified `ext_patch_strategy`, and write `(N, F)` embeddings to HDF5.
*   **Inputs**:
    *   `batch_size` *(int, default=300)*: Number of tiles to pass through the backbone simultaneously.
    *   `num_workers` *(int, default=4)*: Number of multiprocessing workers for reading image patches.
    *   `chunk_size` *(int, default=10000)*: (Tile only) Number of tile embeddings to save per HDF5 chunk.
*   **Outputs**:
    *   *(None)*. Writes directly to `ARTIFACTS/slide_{uuid}/{uuid}_extractions.h5` or `ARTIFACTS/tile_{class}/{class}_extractions.h5`.

### `generate_thumbnails`
**Description**: Generates low-resolution WSI overviews and computes tissue masks.
*   **Inputs**:
    *   `overview_mpp` *(float, optional)*: Resolution for thumbnails. Defaults to config if not provided.
    *   `modes` *(list[str], default=["slide_thumbnail", "masked_thumbnail"])*: Which thumbnails to generate.
*   **Outputs**:
    *   *(None)*. Caches `.png` images to the specific slide's artifact folder.

### `generate_feats_color`
**Description**: Generates UMAP/PCA 3D RGB reductions of the embeddings for visual heatmapping.
*   **Inputs**: *(None)*. Relies on internal configs.
*   **Outputs**: 
    *   *(None)*. Appends `feat_color`, `feat_clusters`, `feat_score` to the `.h5` extractions file.

---

## 3. Analyzer Interfacing

The `Analyzer` is a single-slide/single-class inspector. The Manager "serves" data to it so you can perform targeted queries on a single entity (as seen in notebooks).

### `serve_slide_analyzer` / `serve_tile_analyzer`
**Description**: Loads the cached artifacts for a specific slide or tile class into `self.analyzer` and triggers its internal `load_proc_ext()` to prepare the features and coordinates.
*   **Inputs**:
    *   `slide_id` *(str)*: UUID of the slide.
    *   `tile_class` *(str)*: Directory class name of the tiles.
    *   `data_modes` *(str or list[str], default='req')*: Specifies which artifacts to load from disk into memory. Common values include `'embeddings'`, `'thumbnails'`, `'feats_color'`.
*   **Outputs**:
    *   *(None)*. The `self.analyzer` is now populated. You can subsequently call `self.analyzer.get_slide_thumbnail()`, `self.analyzer.get_feature_map()`, etc.

---

## 4. Training & Data Collection

Functions used to orchestrate ML pipelines via the PyTorch Inferencer framework.

### `get_slide_splits`
**Description**: Automatically creates stratified Train/Validation splits based on the classes present in the configured dataset CSV.
*   **Inputs**:
    *   `df` *(pd.DataFrame)*: The dataset dataframe.
    *   `mode` *(str)*: `'slide'` or `'tile'`.
    *   `id_col` *(str)*: Column name holding the IDs (e.g., `'slide_id'`).
    *   `train_ratio` *(float, default=0.8)*: Fraction of data allocated to training.
    *   `auto_split` *(bool, default=True)*: Uses argmax stratification.
*   **Outputs**:
    *   Returns two sets: `(train_ids, valid_ids)`.

### `intitalise_inferencer`
**Description**: Instantiates the downstream neural network model (e.g. `ROIMIL`, `SparseDETR`) and prepares it for training/evaluation based on project downstream configs.
*   **Inputs**:
    *   `mode` *(str)*: `'slide'` or `'tile'`.
    *   `load_epoch` *(int or str, default='best')*: Which epoch weights to load if a checkpoint exists.
    *   `inf_id` *(int, optional)*: Specific inferencer ID.
*   **Outputs**:
    *   *(None)*. Populates `self.inferencer` (a `Composite` or specific model).

### `train`
**Description**: Executes the PyTorch training loop over `n_epochs`. Calls `slide_features_collector` internally to stream data.
*   **Inputs**:
    *   `n_epochs` *(int, default=100)*: Number of epochs.
    *   `batch_size` *(int, default=1)*: Effective batch size (gradient accumulation steps).
    *   `verbose` *(bool, default=True)*: Print progress.
*   **Outputs**:
    *   Returns a dictionary `train_history` tracking loss over epochs.

### `evaluate`
**Description**: Runs the model in `torch.no_grad()` over the validation set to compute metrics (AUC, F1, Dice).
*   **Inputs**:
    *   `batch_size` *(int, default=1)*.
    *   `eval_ids` *(list or str, default='valid')*: Set of IDs to evaluate.
*   **Outputs**:
    *   Returns a dictionary containing calculated metrics (from `LabelMetrics` or equivalent).

### `slide_features_collector` / `tile_features_collector`
**Description**: A Python generator that seamlessly loads HDF5 embeddings, processes spatial tokens, aligns them with JSON/TIFF annotations, and yields batches of data.
*   **Inputs**:
    *   `collect_ids` *(list)*: IDs to yield.
    *   `shuffle` *(bool, default=False)*: Whether to shuffle the IDs.
    *   `batch_size` *(int)*: Items to yield per iteration.
*   **Yields**:
    *   A dictionary representing the processed WSI/Tile data. Keys typically include:
        *   `id`: UUID
        *   `features`: `(N, F)` tensor
        *   `coords`: `(N, 2)` tensor
        *   `annotation`: `(N, C)` tensor
        *   `label`: `(C,)` tensor
        *   `objects`: `list[dict]` of ROI bags (if supervision is enabled)

---

## 5. Similarity Search (FAISS)

Exposed methods for global descriptor matching (VLAD) via FAISS.

### `fit_from_generator`
**Description**: Dynamically fits the KMeans clustering centroids using a subset of the dataset to avoid out-of-memory errors.
*   **Inputs**:
    *   `descriptor_generator` *(Generator)*: Usually the output of `get_descriptor_generator(mode='slide')`.
    *   `max_samples` *(int, default=250000)*: The absolute maximum number of tiles to ingest into the training buffer.
    *   `max_per_yield` *(int, default=2000)*: Subsamples heavily from each slide to guarantee morphological diversity across the cohort.
*   **Outputs**:
    *   *(None)*. Trains and sets `self.centers` and initializes the `IndexFlatL2`.

### `build_index`
**Description**: Projects all dataset embeddings through the trained centers to generate and store fixed-length `(K * F)` VLAD descriptors.
*   **Inputs**:
    *   `descriptor_generator` *(Generator)*.
*   **Outputs**:
    *   *(None)*. Populates the global FAISS index with the dataset.

### `find_similar_feat` / `find_similar_text`
**Description**: Queries the index to return the most similar slides to a given query.
*   **Inputs**:
    *   `query_feat_vec` / `wrd_emb` *(np.ndarray)*: The feature vector to search for.
    *   `top_k` *(int, default=5)*: Number of matches to return.
*   **Outputs**:
    *   Returns a list of matching `slide_id`s or a tuple containing distances and IDs.

---

## 6. Registry & Database System (`apeiron/manage/registry`)

The `registry` module acts as a lightweight, file-system-backed database system (using Pandas and CSVs) that indexes, queries, and tracks all Whole Slide Images (WSIs) and standalone tiles introduced to the Apeiron environment.

It ensures that every physical WSI or tile folder placed into the environment is assigned a unique UUID. It strictly separates the original raw data from generated machine-learning outputs by mapping raw files in the `DATASETS/` folder to isolated subdirectories in the `ARTIFACTS/` folder.

This module acts as a placeholder for a future PostgreSQL implementation, providing a plug-and-play dataframe interface for the rest of the ecosystem.

---

### Database Architecture
When `Registry` is initialized, it builds out the following directory structure inside your specified `root_dir`:

```text
root_dir/
├── SLIDE_DATABASE/
│   ├── DATASETS/           # You place raw slides here, organized by class (e.g. colon/1.svs)
│   ├── ARTIFACTS/          # Auto-generated by Apeiron
│   │   ├── slide_{uuid1}/
│   │   │   ├── {slide_name}.json                       # Slide Manifest
│   │   │   ├── {slide_name}_extractions.h5             # Feature embeddings
│   │   │   ├── {slide_name}_slide_thumbnail.png        # Generated visuals
│   │   │   └── {slide_name}_masked_thumbnail.png
│   │   └── slide_{uuid2}/...
│   └── registry.csv        # Maps UUID -> SLIDE path & class
└── TILE_DATABASE/
    ├── DATASETS/           # You place standalone tiles here, organized by class
    ├── ARTIFACTS/          # Auto-generated artifacts for tiles
    │   ├── {tile_class_1}/
    │   │   ├── {tile_class}.json                       # Tile Class Manifest
    │   │   ├── {tile_class}_extractions.h5             # Feature embeddings
    │   │   ├── {tile_class}_ext_1.csv                  # Chunk mappings for extraction 1
    │   │   └── {tile_class}_ext_2.csv                  # Chunk mappings for extraction 2
    │   └── {tile_class_2}/...
    └── registry.csv        # Maps UUID -> TILE path & class
```

---

### Core Registry Modules

#### `Registry` (`registry.py`)
The unified entry point that orchestrates the individual slide and tile registries. The `Manager` classes interact almost exclusively with this overarching class.

*   **`ingest_data(data_classes, base_mpps, mode)`**: 
    Scans the requested `data_classes` (folder names under `DATASETS/`) for new files. Depending on the `mode` (`'slide'` or `'tile'`), it delegates to the respective sub-registry to assign UUIDs and update the `registry.csv`.
*   **`slide_entry_generator(slide_ids)`**: 
    Yields data for downstream processors. For a list of `slide_ids`, it yields a tuple containing: `(slide_id, slide_name, slide_path, artifact_folder, artifact_manifest_path, artifact_manifest)`. It automatically creates missing artifact directories.
*   **`tile_entry_generator(tile_ids)`**: 
    Since standalone tiles are usually processed in bulk rather than one-by-one, this yields a *dictionary* grouped by `tile_class`, containing a dataframe of all tile paths in that class, alongside the artifact folder and manifest data.

---

#### `SlideRegistry` (`slide_reg.py`)
Handles WSI-specific tracking.

*   **`ingest_slide_classes()`**: Recursively walks the `DATASETS/` folder looking for valid whole slide extensions (`.svs`, `.ndpi`, `.tif`, etc.).
*   **`_register_slide()`**: Generates a `uuid4().hex` for the slide and logs its path relative to `DATASETS/` to avoid hardcoded absolute paths breaking if the database is moved.
*   **`query_registry()`**: Returns a Pandas DataFrame containing the matched slide UUIDs.

---

#### `TileRegistry` (`tile_reg.py`)
Handles standalone patch/tile tracking. Standalone tiles (like `.png` crops) lack embedded OpenSlide metadata, meaning their physical resolution (Microns Per Pixel, MPP) is lost. `TileRegistry` enforces tracking this.

*   **`ingest_tile_classes(tile_classes, base_mpps)`**: 
    Ingests folders of standalone tiles. Importantly, it pairs every class folder with a `base_mpp` (defaulting to 0.5 if not provided) and physically writes a `base_mpp.txt` into the dataset folder.
*   **`_register_tiles()`**: Handles bulk registration to remain performant when scanning tens of thousands of patch images at once.
*   **MPP Helpers (`_set_tile_class_base_mpp`, `_get_tile_class_base_mpp`)**: Reads/writes the required physical scaling constraints so that down-stream ViT tokenizers or visualizers can scale these standalone tiles correctly as if they were drawn from a WSI.

---

### 🚀 Workflows & Usage

#### 1. Ingestion Examples
To update the registry and assign UUIDs to new files added to `DATASETS/`:

```python
# Assuming you have an `operator` or `manager` instance
operator.ingest_data(mode='slide', data_classes=['samples', 'colon'])
operator.ingest_data(mode='tile', data_classes=['samples', 'lung'])
```
* **Unique UUIDs**: Every unique `path` gets a unique UUID. It is perfectly fine if you have two slides named `001.tiff`, as long as their paths differ (e.g., `samples/001.tiff` vs `colon/001.tiff`).

#### 2. Artifacts & Manifests
Apeiron avoids recalculating embeddings or coordinates by reading from JSON manifests located in the `ARTIFACTS` directories. 

**Slide Manifest (`{slide_name}.json`)**
Tracks thumbnail generation settings and caches feature extractions.
* New `extractions` are appended when configs (`ext_configs`, `coords_configs`, `feats_configs`) do not match existing ones.
* By default, `ext_patch_strategy` is set to `"discard"` for standard models to save disk space. If `"aggr"` or `"keep"` is used, Apeiron will intelligently reuse the saved patch hierarchies instead of re-running the heavy ViT model.

**Tile Manifest (`{tile_class}.json`) & Chunk CSVs**
Since tile classes can contain millions of images, their artifact generation behaves differently:
* Extraction metadata is saved to the class-level JSON.
* The actual embeddings are chunked into `.h5` files (typically 10,000 embeddings per chunk).
* A specific `csv` is generated for each extraction (e.g. `{tile_class}_ext_1.csv`) that acts as a lookup table mapping a specific `tile_id` to its `chunk_id` and `local_id` within the HDF5 file.
* Tiles also support an `ext_type` of `"windowed"` or `"standalone"`. Windowed mode turns each individual `.png` patch into a pseudo-slide to be hierarchically downscaled.

#### 3. Client Project Alignment
When setting up a project workspace outside of Apeiron, you define which slides/tiles to use by referencing the registry's UUIDs in your `dataset.csv`:

```csv
slide_id,slide_name,class0,class1
e8e3e80127da46ed81bb184ca95456ad,001.tiff,1,0.3
e908537e975b407799b6c345574b8be2,002.tiff,0,0.5
```
Apeiron cross-references these `slide_id`s directly against the central `registry.csv` to know exactly which WSI `.svs` or cached `.h5` embeddings to load, allowing multiple independent projects to share the same heavy data pipeline without duplicating embeddings!
