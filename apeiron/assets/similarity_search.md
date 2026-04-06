# VLAD (Vector of Locally Aggregated Descriptors) 

### 🔍 Does VLAD require training?
- **VLAD itself is not trained** in the sense of learning parameters via gradient descent.  
- What *is* required is a **codebook (set of cluster centers)**, typically obtained by running **k-means** on local descriptors (like SIFT, SURF, or CNN features).  
- Once you have those cluster centers, VLAD simply:
  1. Assigns each local descriptor to its nearest cluster center.
  2. Computes the residual (difference between descriptor and center).
  3. Aggregates (sums) residuals per cluster.
  4. Concatenates them into a single vector of dimension \(K \cdot D\) (where \(K\) = number of clusters, \(D\) = descriptor dimension).  

So the “training” is really just **unsupervised clustering** to build the codebook. After that, VLAD is deterministic.

### 📐 Your notation \(N, F \to 1, D\)
- \(N\): number of local descriptors per image.  
- \(F\): dimensionality of each descriptor.  
- VLAD compresses these into a **single global vector** of dimension \(K \cdot F\).  
- That’s why it’s so useful: it turns a variable-length set of local features into a fixed-length representation, enabling efficient similarity search.

### ⚡ Why it works well for similarity search
- VLAD captures **distributional information** about descriptors relative to cluster centers, not just their presence.  
- This makes it more discriminative than simple bag-of-words histograms.  
- Normalization steps (intra-normalization, power normalization, L2) further improve retrieval performance.

👉 In short: VLAD doesn’t “learn” in the deep learning sense — it **summarizes** using a precomputed codebook, which is why it’s lightweight and efficient for feature similarity search.

If you set \(K=1\), then VLAD essentially collapses all \(N\) local descriptors into a single residual vector of dimension \(F\). It’s still a summarization, but with only one “cluster center” the representation is less discriminative. In practice, people use larger \(K\) (like 16, 64, 256) to capture richer structure.  

---

# FAISS (Facebook AI Similarity Search) 

### 🚀 What FAISS does
- **FAISS (Facebook AI Similarity Search)** is a library for **efficient nearest neighbor search** in high-dimensional spaces.  
- It’s not about feature summarization like VLAD — instead, it’s about **indexing and searching** vectors once you already have them.  
- You feed FAISS your global descriptors (VLAD vectors, Fisher vectors, CNN embeddings, etc.), and it builds an index that supports fast similarity queries.  

### 🔧 How VLAD and FAISS fit together
- **VLAD**: turns variable-length local descriptors \((N, F)\) into fixed-length global vectors \((1, K \cdot F)\).  
- **FAISS**: takes those global vectors and allows you to do efficient similarity search across millions of images.  
- Together: VLAD gives you compact, discriminative representations; FAISS makes searching those representations scalable.

### ⚡ Key point
VLAD requires a **codebook (via k-means)** but no supervised training. FAISS requires **index construction** (flat index, IVF, HNSW, PQ, etc.), which is also unsupervised in the sense that it doesn’t learn semantic labels — it just organizes vectors for fast retrieval.

👉 So the workflow is:  
1. Extract local descriptors (e.g., SIFT, CNN features).  
2. Build VLAD vectors using a codebook.  
3. Store VLAD vectors in a FAISS index.  
4. Query FAISS with a new VLAD vector to find nearest neighbors efficiently.

---

# Creating the "Region Score" (The Spatial Step)

A "region" isn't just one high-scoring patch; it's a **cluster** of high-scoring patches. To find these, you need to aggregate the scores based on their $(N, 2)$ coordinates.

#### **A. Spatial Smoothing (The Heatmap)**

Apply a Gaussian kernel or a simple "K-Nearest Neighbors" average over the coordinates.

* **Logic:** For each patch $i$, its new "Region Score" is the average similarity score of all patches within a certain distance $R$ (e.g., 500 microns).
* **Result:** This creates a smooth **Heatmap**. Single outlier patches with high scores get dampened, while groups of patches with moderately high scores "glow" brighter.

#### **B. Density-Based Clustering (DBSCAN)**

To pull out actual "groups" (coordinates of similar regions), you can run **DBSCAN** on the patches that passed a certain similarity threshold.

* **Input:** Coordinates $(x, y)$ of patches where $S_n > 0.8$.
* **Output:** Distinct clusters $C_1, C_2, ... C_k$. These are your "similar regions."

---

### Ranking the Groups

Once you have these clusters, how do you know which one is the "most" similar? You rank them using a **Composite Score**:

1. **Average Similarity:** The mean cosine similarity of the patches inside the cluster.
2. **Cluster Mass:** The total number of patches (size of the region).
3. **Spatial Configuration:** (Advanced) If you want to be fancy, you compare the "shape" of the cluster to your query $Q$.
