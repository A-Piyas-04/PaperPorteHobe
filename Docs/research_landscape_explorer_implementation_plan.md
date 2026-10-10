> **Documentation status (2026-10-10):** Historical implementation sequence, retained for project context. For current development use [Local development](next-instructions.md).

# Research Landscape Explorer — Step-by-Step Implementation Plan

## 1. Set Up the Project
- Create the GitHub repository.
- Create these folders:
  - `notebooks/`
  - `app/`
  - `data/processed/`
  - `configs/`
  - `reports/`
- Create a Python environment.
- Install the required libraries.

## 2. Get and Clean the arXiv Data
- Download the recent arXiv metadata dataset.
- Keep only `cs.*` papers.
- Use a fixed recent time range, e.g. 2023 onward.
- Keep fields such as:
  - arXiv ID
  - Title
  - Abstract
  - Category
  - Date
- Remove empty or very short abstracts.
- Remove duplicates.

## 3. Generate Embeddings
- Combine `title + abstract`.
- Use `all-MiniLM-L6-v2`.
- Generate embeddings for all selected papers.
- Save the embeddings immediately so they do not need to be recomputed.

## 4. Create Research Clusters
- Reduce embeddings to around **5–10 dimensions using UMAP**.
- Run **HDBSCAN** on the reduced embeddings.
- Store each paper's cluster ID.
- Keep noise/unclustered papers instead of forcing everything into a cluster.

## 5. Generate Cluster Labels
- Use **c-TF-IDF / BERTopic-style keywords**.
- Save representative paper titles for each cluster.
- Use the keywords and representative papers to describe each cluster.

## 6. Create the 2D Research Map
- Run a separate **2D UMAP** for visualization.
- Plot papers using Plotly.
- Group or color papers by cluster.
- Add hover information such as:
  - Title
  - Cluster
  - Date
  - Category

## 7. Add Research Growth Analysis
- Count papers per cluster over time.
- Compare a recent time window with a previous time window.
- Calculate normalized `relative_growth`.
- Also show raw monthly publication counts.

## 8. Add Semantic Search
- Take the user's search query.
- Embed it using the same embedding model.
- Find nearest papers using cosine similarity.
- Highlight those papers on the map.
- Show which clusters they belong to.

## 9. Add the Sparse-Neighborhood Feature
- Detect low-density areas near relevant or growing clusters on the 2D map.
- Check those areas again in higher-dimensional embedding space.
- Show nearby papers as evidence.
- Keep only signals that remain reasonably stable across different UMAP runs.

## 10. Validate the System
- Manually inspect papers inside major clusters.
- Test around 10–20 semantic-search queries.
- Compare growth using 3-, 6-, and 12-month windows.
- Test multiple UMAP settings or seeds.
- Record failures and unstable results.

## 11. Build the Streamlit/Gradio App
Include:
- Research landscape/map
- Search bar
- Cluster details
- Growth chart
- Sparse-neighborhood toggle
- Evidence papers
- Methodology and limitations

## 12. Optional: Add Citation Missing-Link Analysis
After the main system works:
- Get citation relationships from OpenAlex.
- Compare semantic similarity with citation connections.
- Find clusters or papers that are semantically close but weakly connected by citations.
- Show these as **potential missing connections**, not proven research gaps.

## 13. Deploy
- Precompute:
  - Embeddings
  - Clusters
  - UMAP coordinates
  - Growth statistics
- Keep the online app lightweight.
- Deploy using **Hugging Face Spaces**.
- Push the complete reproducible pipeline to GitHub.

---

## Recommended Implementation Order

**Data → Embeddings → Clustering → Labels → 2D Map → Growth → Search → Sparse Signal → Validation → App → Citation Feature → Deploy**
