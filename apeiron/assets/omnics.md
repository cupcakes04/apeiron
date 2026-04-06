# Omnics Modality

Here’s a compact cheat‑sheet you can keep in mind when thinking about omics pathology in data science and AI:

---

## 🧬 Omics Data Types
- **Genomics** → DNA mutations, copy number changes.  
- **Transcriptomics (RNA‑seq)** → Gene expression (blueprints from DNA).  
- **Proteomics** → Protein abundance/modifications.  
- **Metabolomics** → Small molecules/metabolic activity.  
- **Spatial Transcriptomics** → Gene expression mapped directly onto tissue architecture.

---

## 📊 Data Structures (CSV Mental Model)
- **Rows** = samples (patients, cells, tissue spots).  
- **Columns** = features (genes/proteins/metabolites).  
- **Diagnosis/Target column** = label (disease subtype, cell type, spatial region).  

Example:
```csv
SampleID,Target,Gene1,Gene2,Gene3,...
S001,Cancer,120,45,0,...
S002,T-cell,15,200,5,...
S003,Neuron,5,0,500,...
```

---

## 🧩 Pathology AI Targets
- **Bulk RNA‑seq** → whole tissue average; target = disease subtype.  
- **Single‑cell RNA‑seq** → per‑cell profiles; target = cell type.  
- **Spatial transcriptomics** → per‑spot/cell with coordinates; target = spatial + cell type.  

---

## 🔗 Integration with Pathology Images
- **Bulk** → downstream correlation (not direct integration).  
- **Single‑cell** → cell identity prediction, but no spatial context.  
- **Spatial** → direct multimodal fusion (visual + gene expression) for foundation models.  

---

## ⚙️ Preprocessing Essentials
- Normalization (log‑scaling, TPM, CPM).  
- Batch correction (remove technical noise).  
- Missing data handling.  
- ID mapping (consistent gene/protein identifiers).  

---

## 🚧 Challenges
- **High dimensionality**: thousands of genes vs few samples.  
- **Batch effects**: lab/tech variability.  
- **Integration complexity**: multiple omics layers.  
- **Clinical translation**: interpretability and reproducibility.  

---

👉 Think of it this way:  
- **Bulk** = “average voice of the tissue.”  
- **Single‑cell** = “each cell’s voice.”  
- **Spatial** = “voices placed on a map.”  

---

Here’s a **cheat‑sheet table** comparing bulk, single‑cell, and spatial transcriptomics in pathology AI — so you can quickly glance at the differences:

| 🧬 Data Type            | 📊 Structure (CSV mental model) | 🎯 Target/Label Examples | 🔗 Integration with Pathology Images | 💡 Key Notes |
|--------------------------|---------------------------------|--------------------------|--------------------------------------|--------------|
| **Bulk RNA‑seq**         | Rows = patients/tissue samples<br>Cols = average gene expression across all cells | Disease subtype (e.g., cancer vs normal) | Indirect — used downstream to correlate with whole‑slide image predictions | Cheap, but loses cell‑level and spatial detail |
| **Single‑cell RNA‑seq**  | Rows = individual cells<br>Cols = per‑cell gene expression | Cell type (T‑cell, B‑cell, neuron, etc.) | Limited — morphology vs expression, but no spatial coordinates | Great for cell identity, but expensive and complex |
| **Spatial transcriptomics** | Rows = tissue spots/cells with coordinates<br>Cols = gene expression per spot | Cell type + spatial region (tumor vs stroma, microenvironment) | Direct — can align gene maps with histology images for multimodal AI | Most powerful for foundation models, but costly |

---

### 🧾 Quick Reminders
- **Diagnosis column** can mean *disease*, *cell type*, or *spatial region* depending on experiment.  
- **Gene expression values** are usually RNA counts (normalized/log‑scaled).  
- **Integration potential** is highest with spatial transcriptomics, moderate with single‑cell, and lowest with bulk.  

👉 Think of it as:
- Bulk = “average voice of the tissue.”  
- Single‑cell = “each cell’s voice.”  
- Spatial = “voices placed on a map.”  

---