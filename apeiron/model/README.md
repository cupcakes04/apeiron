# APEIRON Modeling Pipeline

The `apeiron/model` directory contains the core neural network architectures for APEIRON. The framework is designed around a **multi-modal, multi-head** paradigm where a shared foundational **Backbone** extracts representations that are then processed by a composite set of task-specific **Downstream** models.

## Architecture Pipeline

1. **Backbone**: Frozen or LoRA-tuned Vision Transformers (e.g., UNI, CONCH, Virchow) that extract patch and tile-level representations.
2. **Inferencer**: The orchestrator that takes raw features, builds the `MultiHeadModel`, routes inputs to the correct downstream heads, manages the `MultiModalLoss`, and computes metrics using `MultiModalPredict`.
3. **Downstream**: The individual task heads. All downstream models take features of shape `(B, N, F)` where `B` is batch size, `N` is sequence length, and `F` is the feature dimension.

## Downstream Models

The downstream modeling ecosystem is split into distinct paradigms that operate simultaneously over the same shared features. This allows a single pass of features to predict slide-level classes, tile-level segmentations, detect objects, and generate text descriptions.

### `MultiHeadModel` (`downstream/unify.py`)
This is the master container. It instantiates the correct combination of models based on the `inf_models` argument (e.g., `['abmil', 'unet']`).
- **Input**: `features (B, N, F)`, optional `text (B,)` or `list[str]`
- **Output**: `ModelData` object containing logits, attention scores, object boxes, and embeddings across all active heads.

### 1. Classification (CLS)
- **`mlp`**: Simple Multi-Layer Perceptron. Averages `(B, N, F)` into `(B, F)` and projects to classes.
- **Input Shape**: `(B, N, F)` -> **Output Shape**: `(B, C)`

### 2. Multiple Instance Learning (MIL)
Models designed to aggregate variable-length bags of instances (tiles) into a single slide-level prediction, while computing attention scores for interpretability.
- **`abmil`**: Attention-Based MIL (Ilse et al.). Computes a scalar attention score for each instance, computes a weighted average of features, and projects to classes.
- **`transmil`**: Correlated MIL using a Transformer encoder layer. Adds positional embeddings (PPEG) and self-attention before aggregating.
- **Input Shape**: `(B, N, F)` -> **Output Shape**: 
  - Logits: `(B, C)`
  - Attention: `(B, N, 1)`

### 3. Segmentation (SEG)
Models that assign a class label to every individual token/tile in the sequence. If the input is a grid of tiles or patches, these act as dense prediction networks.
- **`unet`**: 1D U-Net operating on the sequence dimension.
- **`fpn`**: Feature Pyramid Network over the sequence.
- **Input Shape**: `(B, N, F)` -> **Output Shape**: `(B, N, C)`

### 4. Vision-Language Modeling (VLM)
Models designed to align histology image features with text or generate text from them.
- **`gen_vlm`**: Generative VLM. Uses a Perceiver Resampler to compress `(B, N, F)` into a fixed number of visual tokens `(B, K, H)`, concatenates them with text embeddings, and uses a causal language model (e.g., Llama) to generate text autoregressively.
  - **Output**: Generative Cross-Entropy Loss, Text Generation function.
- **`con_vlm`**: Contrastive VLM. Aligns image features and text embeddings using an InfoNCE loss (CLIP-style). Uses a frozen text encoder (e.g., BioBERT/PubMedBERT).
  - **Output**: Contrastive InfoNCE Loss, Embedding functions.

### 5. Object Detection (OBJ)
Models that predict bounding boxes and class labels from a set of features.
- **`detr`**: Transformer-based detection head (inspired by DETR). Uses learned object queries and cross-attention over the `(B, N, F)` features to predict bounding boxes and classes.
- **Input Shape**: `(B, N, F)` -> **Output Shape**:
  - Boxes: `(B, Num_Queries, 4)`
  - Classes: `(B, Num_Queries, C + 1)` (includes 'no-object' class)

## Multi-Modal Loss & Prediction

Because APEIRON runs multiple heads simultaneously, it requires a unified system to handle the loss and metric calculations for all branches.

- **`MultiModalLoss` (`downstream/loss/unify.py`)**: Computes the specific loss for each branch (e.g., Hard Cross-Entropy for `abmil`, Dice Loss for `unet`, Bounding Box L1 + GIoU for `detr`). It then wraps all losses using `AutomaticWeightedLoss` (AWL) to dynamically balance the gradients during training.
- **`MultiModalPredict` (`inference/predictor.py`)**: Takes the raw logits from `MultiHeadModel`, applies the appropriate activation (Sigmoid/Softmax), computes discrete predictions based on thresholds, and calculates branch-specific metrics (Accuracy, AUROC, F1, Dice, mAP).

## Data Flow Summary

```
1. Image (WSI / Tiles)
       │
       ▼
2. Backbone (e.g., H-optimus-0) -> Extractor -> (N, F) features
       │
       ▼
3. Batching / Dataloader -> (B, N, F)
       │
       ▼
4. MultiHeadModel (Inferencer)
       │
       ├──> ABMIL -> (B, C) slide logits, (B, N, 1) attention
       ├──> UNET  -> (B, N, C) tile-level logits
       └──> VLM   -> Generative Loss / Contrastive Embeddings
       │
       ▼
5. MultiModalLoss -> Dynamically weighted composite loss
       │
       ▼
6. MultiModalPredict -> ModelData (Losses, Metrics, Predictions)
```
