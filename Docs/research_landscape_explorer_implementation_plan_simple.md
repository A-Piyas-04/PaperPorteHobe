# Research Landscape Explorer — Implementation Plan

1. **Create the GitHub repo and project structure**
   - Set up:
     - `notebooks/`
     - `app/`
     - `data/processed/`
     - `configs/`
     - `reports/`
   - Create the Python environment.
   - Add `.gitignore` and `requirements.txt`.

2. **Get and prepare the arXiv dataset**
   - Download recent arXiv metadata.
   - Keep only `cs.*` papers.
   - Choose a fixed recent time range.
   - Remove empty, very short, and duplicate records.
   - Save the cleaned dataset.

3. **Generate paper embeddings**
   - Combine `title + abstract`.
   - Use `all-MiniLM-L6-v2`.
   - Generate embeddings for all selected papers.
   - Save embeddings so they do not need to be recomputed.

4. **Create research clusters**
   - Reduce embeddings to about 5–10 dimensions using UMAP.
   - Run HDBSCAN.
   - Assign each paper a cluster ID.
   - Keep noise/unclustered papers.

5. **Label the clusters**
   - Generate representative keywords using c-TF-IDF / BERTopic-style methods.
   - Store representative paper titles for each cluster.
   - Use them to describe each cluster.

6. **Create the 2D research landscape**
   - Generate a separate 2D UMAP for visualization.
   - Plot papers with Plotly.
   - Group/color papers by cluster.
   - Add useful hover information.

7. **Add publication-growth analysis**
   - Count papers per cluster over time.
   - Compare recent and previous time windows.
   - Calculate normalized relative growth.
   - Also show raw monthly publication counts.

8. **Add semantic search**
   - Take the user's research query.
   - Embed it with the same embedding model.
   - Find nearest papers using cosine similarity.
   - Highlight the nearest papers on the map.
   - Show which clusters they belong to.

9. **Add sparse-neighborhood analysis**
   - Detect low-density areas near relevant or growing clusters.
   - Verify them in higher-dimensional embedding space.
   - Show nearby papers as evidence.
   - Keep only reasonably stable signals.
   - Do not present them as proven research gaps.

10. **Validate the ML results**
    - Inspect major clusters manually.
    - Test semantic search using known queries.
    - Compare growth using different time windows.
    - Test multiple UMAP seeds/settings.
    - Record unstable or failed cases.

11. **Build the web app**
    - Use Streamlit or Gradio.
    - Include:
      - Research landscape
      - Search
      - Cluster details
      - Growth chart
      - Sparse-neighborhood exploration
      - Evidence papers
      - Methodology
      - Limitations

12. **Add citation missing-link analysis**
    - Do this after the main system works.
    - Get citation relationships from OpenAlex.
    - Compare semantic similarity with citation connectivity.
    - Find semantically close but weakly connected papers/clusters.
    - Present them as potential missing connections.

13. **Optimize for deployment**
    - Precompute:
      - Embeddings
      - Cluster assignments
      - UMAP coordinates
      - Growth statistics
      - Other expensive analytical artifacts
    - Keep the deployed app lightweight.

14. **Deploy and finish the repository**
    - Deploy the app to Hugging Face Spaces.
    - Complete the README.
    - Document methodology and limitations.
    - Add validation results.
    - Ensure the pipeline can be reproduced from the repository.

## Implementation Order

**Repo → Data → Embeddings → Clusters → Labels → Map → Growth → Search → Sparse Analysis → Validation → App → Citation Analysis → Deploy**
