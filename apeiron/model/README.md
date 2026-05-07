# Apeiron Model API Reference (`apeiron/model`)

## Overview
The `apeiron/model` directory houses the deep learning architectures. It is distinctly split into three functional layers:

1. **`backbone`**: Heavy, pre-trained Vision Transformers (ViTs) that extract high-dimensional embeddings from raw pixels (e.g., UNI, CONCH, H-optimus-0).
2. **`downstream`**: Task-specific neural network heads (e.g., ABMIL, DETR, UNet, Classifiers) that accept pre-extracted embeddings to predict diagnostic labels or annotations.
3. **`inference`**: The orchestrating training framework. It wraps the downstream models into a `MultiHeadModel`, handles gradients via the `Inferencer`, and defines the strict tensor typing interfaces (`utility.py`).

---

## 1. Backbone (`apeiron/model/backbone`)

### `Backbone` (`backbone.py`)
**Description**: The model registry and loader for Foundation Models. It ensures that massive ViTs are downloaded, cached locally, and loaded into GPU memory efficiently.
*   **Initialization Inputs**: 
    *   `models_save_dir` *(str or Path)*: Directory to cache `.pth` weights.
    *   `device` *(str, optional)*: `'cuda'` or `'cpu'`. Auto-detects if None.
*   **`select_model(model_name)`**:
    *   **Inputs**: `model_name` *(Literal)*: Must be one of `['hop0', 'hop1', 'vir1', 'vir2', 'ch15', 'uni2h', 'mstar', 'dino3']`.
    *   **Outputs**: Changes internal state to the requested model path.
*   **`create_model(model_name=None, zero_shot=False)`**:
    *   **Inputs**: 
        *   `model_name` *(str, optional)*: Overrides selected model if provided.
        *   `zero_shot` *(bool, default=False)*: If True, uses text-aligned contrastive weights (if the model supports it) instead of plain image weights.
    *   **Description**: Physically loads the model weights into `self.model` and attaches the image transforms to `self.transform`. It automatically wraps models with `FmWrappers` to ensure *all* models standardize their outputs to return both `[CLS]` tokens and spatial patch tokens.

### `Extractor` (`extract.py`)
**Description**: Inherited by `Analyzer`, it acts as the interface to stream arrays of pixels through the `Backbone`.
*   **`extract_tiles(batch_size=300, num_workers=4, ext_patch_strategy="discard")`**:
    *   **Inputs**: 
        *   `batch_size` *(int, default=300)*: Max arrays processed simultaneously on GPU.
        *   `num_workers` *(int, default=4)*: Multi-processing data loaders.
        *   `ext_patch_strategy` *(Literal)*: Must be one of `["aggr", "discard", "keep"]`.
            *   `'discard'`: Only keep class tokens (lowest memory, shape `N×F`).
            *   `'keep'`: Keep full spatial hierarchy including all patch tokens (shape `N×P×F`).
            *   `'aggr'`: Applies maximum/mean pooling across patch tokens (shape `N×F`).
    *   **Outputs**: Computes and stores the final embeddings and grid coordinates into `self.proc_ext`.

---

## 2. Downstream Models (`apeiron/model/downstream`)

All downstream models act strictly as PyTorch `nn.Module` classes. They do not handle raw pixels. They exclusively accept pre-computed `(N, F)` or `(B, N, F)` feature embeddings.

### Architecture Contract (`__init__.py`)
Every downstream model (e.g., `ABMIL`, `ROIMIL`, `SparseDETR`, `UNet`) must conform to the following API so `Inferencer` can hook into them automatically:

*   **`forward(self, features: torch.Tensor, **kwargs) -> dict`**:
    *   **Input**: `features` `(B, N, F)`. Can also receive `objects` (ROI bags) or `text` via kwargs.
    *   **Output**: A dictionary mapping to `HeadData` keys (e.g. `lbl_logits`, `ann_logits`, `obj_classes`, `attention`).
*   **`loss(self, **kwargs) -> dict`**:
    *   **Input**: Ground truth targets (`label`, `annotation`, `objects`).
    *   **Output**: A dictionary mapped to objective categories (e.g. `{'label': {'lbl_loss': tensor(0.5)}}`).
*   **`predict(self, **kwargs) -> dict`**:
    *   **Output**: Returns post-processed detached Numpy predictions (e.g., `pred_lbl` probabilities, `pred_obj` bounding dictionaries with masks).
*   **`metric(self, threshold=0.5, **kwargs) -> dict`**:
    *   **Output**: Computes F1, AUC, Dice, etc. (e.g., `{'label': {'auc': 0.85}}`).

---

## 3. Inference Engine (`apeiron/model/inference`)

This module manages PyTorch training loops, dynamic loss balancing, and data structures.

### Data Structures (`utility.py`)
Defines strictly typed PyTorch dataclasses passed between the Data `Collector` and the Neural Network. These essentially catch the `PreData` from the `entity` module and store the downstream network's transformations over it.

#### 1. `HeadData`
Stores the raw, unnormalized network outputs (logits and raw attention) directly from the `forward()` pass. These remain on the GPU until explicitly moved.
*   `attention`: `(B, 1, N)` or `(B, C, N)` unnormalized attention scores.
*   `lbl_logits`: `(B, C)` slide-level class logits.
*   `ann_logits`: `(B, N, C)` tile-level annotation logits.
*   `obj_classes`: `(B, Q, C+1)` per-query class logits (includes 'no-object' class).
*   `obj_masks`: `(B, Q, N)` per-query binary mask logits.
*   `vis_emb` / `img_emb`: `(B, H)` pooled visual or image embeddings.

#### 2. `PredData`
Stores post-processed predictions (Sigmoid/Softmax applied) that have been detached from the computational graph and pushed to the CPU as NumPy arrays.
*   `pred_atn`: `(B, 1, N)` normalized attention weights (e.g., via Softmax).
*   `pred_lbl`: `(B, C)` final slide-level probabilities.
*   `pred_ann`: `(B, N, C)` final tile-level probabilities.
*   `pred_obj`: A `list` (length `B`) of lists containing dictionaries for each detected ROI: `{"ids": int (the roi coords id), "labels": length C, "scores": float}`.
*   `pred_txt`: A `list[str]` of generated text strings.
*   **Post-processing fields**: `pred_crd` (coordinates), `pred_scr` (scores), and `pred_data_type` used by the Visualiser to render overlays.

#### 3. `Objectives`
A container mapping raw PyTorch loss or metric tensors to their specific objective categories.
*   `label`: Slide-level objectives (e.g. `{'lbl_loss': tensor(0.5)}` or `{'auc': 0.85}`).
*   `annotation`: Tile-level objectives (e.g. `{'ann_loss': tensor(0.3)}`).
*   `objects`: ROI-level detection objectives (e.g. `{'obj_loss': tensor(1.2)}`).
*   `text`: Text-generation objectives (e.g. `{'gen_loss': tensor(2.1)}`).

#### 4. `ModelData`
The unified state container holding `head`, `pred`, `loss`, and `metric` during a training step.
*   **`assign(mode, **kwargs)`**: Automatically routes dictionaries to the correct sub-dataclass. For example, `assign('head', lbl_logits=x)` places `x` directly into `self.head.lbl_logits`.
*   **`composite_loss(final_loss, **importance)`**: Caches the auto-weighted `final_loss` used for `backward()` alongside the homoscedastic uncertainty importances.

---

### `MultiHeadModel` (`composite.py`)
**Description**: A dynamic `nn.Module` that accepts multiple `inf_models`. It routes the input embeddings through a `SharedEncoder` (to project raw backbone dimensions, e.g., 1536 -> 256), and then parallelizes the features across multiple independent downstream heads (e.g., combining `abmil` + `detr` in one pass).
*   **Initialization Inputs**:
    *   `in_features` *(int)*: Dimension of input backbone embeddings (e.g. 768 or 1536).
    *   `mode` *(str, optional)*: `'slide'` or `'tile'`.
    *   `inf_models` *(str or list[str], default='abmil')*: A list of models chosen from `['abmil', 'gatmil', 'roimil', 'classifier', 'unet', 'spconv', 'detr', 'generative', 'contrastive']`.
    *   *Also passes all `kwargs` (like loss types and classes) down to `choose_inferencer()`*.
*   **`forward(features, **kwargs)`**: Routes data to all sub-heads and compiles their outputs into a single `ModelData` object.
*   **`AutomaticWeightedLoss`**: Dynamically weights multi-task losses using learnable variances (homoscedastic uncertainty) so you do not have to manually tune loss ratios between Slide prediction and ROI detection.

### `Inferencer` (`inferencer.py`)
**Description**: The frontend controller. The `Collector` uses this class to run training epochs. It handles GPU data transferring, optimizer stepping, and evaluation.
*   **Initialization Inputs**: `device` *(str, optional)*: Explicitly define `'cuda'` or `'cpu'`.
*   **`setup_inferencer(...)`**:
    *   **Description**: Initializes the `MultiHeadModel`, sets up the PyTorch optimizers, and prepares the loss criterions. 
    *   **Inputs**:
        *   `in_features` *(int)*: Backbone feature size.
        *   `inf_models` *(str or list[str], default='abmil')*: Head(s) to initialize.
        *   `mode` *(str, optional)*: `'slide'` or `'tile'`.
        *   `lbl_n_classes` / `ann_n_classes` *(int, optional)*: Number of target classes for slides / tiles.
        *   `lbl_loss_type` / `ann_loss_type` *(str, default='bce')*: Objective functions (`'bce'`, `'hard_ce'`, `'mse'`).
        *   `lbl_cls_weights` / `ann_cls_weights` *(dict, optional)*: Class imbalance weighting.
        *   `lr` *(float, default=1e-4)*: Learning rate.
        *   `optimizer` *(str, default='adam')*: Type of optimizer.
        *   `weight_decay` *(float, default=0.0)*: Weight decay / L2 Penalty.
        *   `scheduler` *(str, optional)*: Learning rate scheduler type.
        *   `chkp_path` *(str, optional)*: Preload weights if path is provided.
*   **`train_epoch(data_collector)`**:
    *   **Inputs**: `data_collector` *(Generator)* yielding dictionaries of `(features, label, annotation)`.
    *   **Outputs**: Accumulates gradients across the batch size, runs `optimizer.step()`, and returns aggregated metrics/loss dicts.
*   **`eval_epoch(data_collector, threshold=0.5)`**:
    *   **Inputs**: `data_collector` *(Generator)* and `threshold` *(float, default=0.5)* for binarizing sigmoid predictions.
    *   **Outputs**: Returns a tuple `(epoch_losses, epoch_metrics, predictions)`.
*   **`predict_data(data, threshold=0.5, run_metric=True)`**:
    *   **Description**: A quick single-sample forward pass. Moves `data` dict to CUDA, runs inference, moves predictions back to CPU, and yields the `ModelData` result.
*   **`save_inferencer(chkp_path, epoch=None)` / `load_inferencer(chkp_path)`**: Safely saves/loads optimizer states and model `.pth` dicts.
