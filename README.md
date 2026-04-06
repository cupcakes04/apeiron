# APEIRON

A computational pathology framework for whole slide image (WSI) and tile-based analysis. APEIRON provides end-to-end tools for reading, tiling, feature extraction, visualization, annotation, dataset management, and highly-customizable multi-modal downstream inference for histopathology workflows.

## Architecture Overview

APEIRON is organized into two main levels:

- **Analyzer** — Low-level worker for single-slide or single-tile analysis. Handles slide reading, tiling, feature extraction, post-processing, annotation, similarity search execution, and single-batch downstream inference.
- **Operator** — High-level interface that connects the Registry, Manager, Collector, and Searcher for batch workflows, training, and database management.

Both share a single **Backbone** instance that manages foundational model loading, caching, and selection. Downstream modeling leverages **Inferencer** to construct arbitrary multi-head architectures.

### Supported Foundational Models

| Key    | Model         | Source       | Feature Dim |
|--------|---------------|--------------|-------------|
| `hop0` | H-optimus-0   | Bioptimus    | 1536        |
| `hop1` | H-optimus-1   | Bioptimus    | 1536        |
| `vir1` | Virchow       | Paige AI     | 1280        |
| `vir2` | Virchow2      | Paige AI     | 1280        |
| `ch15` | CONCH 1.5     | Mahmood Lab  | 512         |
| `uni2h`| UNI2-h        | Mahmood Lab  | 1024        |
| `mstar`| mSTAR         | Wangyh       | 768         |
| `dino3`| DINOv3        | Meta         | 1280        |

---

## Quick Start

### 1. Analyzer (Low-Level)

The `Analyzer` operates on a single slide or tile set at a time.

```python
import apeiron as ap

# 1. Setup
backbone = ap.Backbone(models_save_dir="/path/to/models/foundational_models")
analyzer = ap.Analyzer(backbone, ext_enc=224, ext_mpp=0.5)

# 2. Assign a foundational model
analyzer.assign_model(model_name='hop0')

# 3. Open a slide
analyzer.open_slide(slide_path='/path/to/slide.svs')
```

#### Thumbnails & Normalization

```python
# RGB thumbnail
analyzer.get_slide_thumbnail(target_mpp=8.0)

# Binary tissue mask
analyzer.get_masked_thumbnail(target_mpp=8.0)

# Stain normalization
out_img = analyzer.normalise_image(
    analyzer.slide_thumbnail,
    norm_configs={"target_img_path": "/path/to/target.png", "method": "macenko"}
)
```

#### Tile Coordinate Generation & Extraction

```python
# Generate tile coordinates from tissue mask
analyzer.create_tile_coords(method='morphological', tile_threshold=0.25, stride=0.0)

# Prepare dataset and extract features
analyzer.prepare_tiles_dataset(mode='slide')
analyzer.extract_tiles(batch_size=300, num_workers=4, ext_patch_strategy='keep')
```

`ext_patch_strategy` controls how patch tokens are handled:
- `'discard'` — Only keep class tokens (lowest memory, N×F).
- `'aggr'` — Aggregate patches via max/mean pooling (medium memory, N×2F).
- `'keep'` — Keep all patch tokens (highest memory, N×256×F).

#### Feature Post-Processing & Visualization

```python
# Convert embeddings to features at a spatial resolution
analyzer.prepare_features(window_level='tile', patch_to_tile='mean', grid_size=2)

# Dimensionality reduction to RGB (PCA or UMAP)
analyzer.compute_feats_color(method='pca')
analyzer.create_feature_image(mode='color')
analyzer.visualise_overlay(alpha=0.5)

# K-means clustering
analyzer.compute_feats_color(n_clusters=10)
analyzer.create_feature_image(mode='clusters')
analyzer.visualise_overlay(alpha=0.5)
```

#### Single-Slide Inference

```python
# Set up a downstream model (e.g. ABMIL for slide classification)
analyzer.prepare_inferencer(
    mode='slide', 
    feats_configs={'window_level': 'tile', 'patch_to_tile': 'mean'},
    inf_models='abmil',
    lbl_n_classes=2,
    lbl_loss_type='hard_ce'
)

# Run inference and format predictions
model_data = analyzer.predict(mode='slide', batch_size=300)
print(model_data.pred.pred_lbl)  # Slide-level class predictions
```

---

### 2. Operator (High-Level)

The `Operator` connects the Registry, Manager, Collector, and Searcher for batch workflows, training, and database management.

```python
import apeiron as ap

root_dir = "/path/to/DATABASE"
project_path = "/path/to/PROJECTS/user1/task1"
models_save_dir = "/path/to/MODELS/foundational_models"

backbone = ap.Backbone(models_save_dir)
operator = ap.Operator(backbone=backbone, root_dir=root_dir, project_path=project_path)
operator.setup(project_path)
```

#### Batch Generation

```python
# Generate thumbnails, embeddings, and visualization features for all selected slides
operator.generate_thumbnails(modes=["slide_thumbnail", "masked_thumbnail"])
operator.generate_embeddings_slide(batch_size=300, num_workers=4)
operator.generate_feats_color()
```

#### Serving & Interaction

```python
# Serve a pre-configured analyzer with all generated data loaded instantly
slide_id = operator.lookup_table('TCGA-2A-A8VV-01Z-00-DX1', mode='slide')[0]
analyzer = operator.serve_slide_analyzer(slide_id, data_modes='all')
```

#### Similarity Search (FAISS + VLAD)

APEIRON uses Vector of Locally Aggregated Descriptors (VLAD) on GPU-accelerated FAISS to compress arbitrary numbers of slide patches into single fixed-length global embeddings for rapid search.

```python
# Search for similar whole-slide features (global) or regions of interest (ROI)
results = operator.similarity_search(
    mode='slide',
    query_mode=['feat', 'roi'], 
    query_feat_id=[slide_id],
    query_roi_id=[10, 11, 12, 13],  # Tile indices within the query slide
    top_k=5,
    similarity_threshold=0.85
)

print(results.feat_res)  # Global slide similarity DataFrame
print(results.roi_res)   # Local patch/region similarity DataFrame
```

#### Multi-Modal Downstream Training

APEIRON's `Composite` architecture allows combining Classification, MIL, Segmentation, Detection, and Vision-Language Modeling (VLM) seamlessly behind a shared encoder.

```python
# 1. Initialize the dataset collector
operator.intitalise_inferencer(mode='slide')

# 2. Configure a multi-head downstream model
# Example: Slide-level MIL + Tile-level Segmentation + Vision-Language contrastive learning
configs = operator.analyzer.prepare_inferencer(
    mode='slide',
    feats_configs=operator.collector.slide_feats_configs,
    inf_models=['abmil', 'unet', 'convl'],  # Multi-head model
    lbl_class_id_map={'normal': 0, 'tumor': 1},
    ann_class_id_map={'stroma': 0, 'epithelium': 1},
    lbl_loss_type='hard_ce',
    ann_loss_type='soft_ce',
    return_cfgs=True
)

operator.analyzer.setup_inferencer(**configs)
operator.analyzer.setup_optimizer(lr=1e-4, optimizer='adam')

# 3. Train using data generators
for epoch in range(10):
    train_metrics = operator.train(epoch, batch_size=1)
    val_metrics = operator.evaluate(epoch, batch_size=1)
```

APEIRON automatically routes data correctly, handles loss weighting via `AutomaticWeightedLoss`, and populates a unified `ModelData` dataclass.

---

## Directory Structure

### Database Layout

```
DATABASE/
├── SLIDE_DATABASE/
│   ├── DATASETS/
│   │   ├── colon/
│   │   └── prostate/
│   ├── ARTIFACTS/
│   │   ├── slide_{uuid}/
│   │   │   ├── {slide_name}.json              # Artifact manifest
│   │   │   ├── {slide_name}_extractions.h5    # Embeddings & features
│   │   │   └── {slide_name}_slide_thumbnail.png
│   │   └── ...
│   └── registry.csv
│
└── TILE_DATABASE/
    ├── DATASETS/
    ├── ARTIFACTS/
    └── registry.csv
```

### Project Layout

```
PROJECTS/
└── task1/
    ├── config.yaml            # Project configuration
    ├── slide_dataset.csv      # Slide dataset with text/labels
    ├── normalise_targets/     # Stain normalization targets
    └── annotations/
        ├── shape/             # JSON polygon/ellipse annotations
        └── pixel/             # Binary/multi-class mask annotations
```

---

## Configuration

### config.yaml

Place a `config.yaml` in your project directory to define extraction and downstream paradigms:

```yaml
overview_mpp: 8.0

slide_ext:
  ext_configs:
    ext_enc: 224             # Encoder window size (pixels)
    ext_mpp: 0.5             # Extraction resolution
    ext_model: hop0          # Foundational model key
    ext_patch_strategy: discard
  coords_configs:
    method: morphological    
    ann_mask: false           
    tile_threshold: 0.25     
    stride: 0.0              
  feats_configs:
    window_level: tile       # 'grid', 'tile', 'patch'
    patch_to_tile: discard   
    grid_size: 2             

slide_gt:
  ann_configs:
    ann_type: shape          # 'shape' (JSON) or 'pixel' (mask)
    supervision: false          # Bag annotation regions as 'pseudo-slide'
    class_id_map:
      '0': background
      '1': tumor
  label_configs:
    label_id_map:
      '0': background
      '1': tumor
```
