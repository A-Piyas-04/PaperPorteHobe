# Notebooks

The analytical logic lives in the reusable `scholargrid/` package and is
orchestrated by [`pipeline/run_pipeline.py`](../pipeline/run_pipeline.py), which
keeps the offline computation reproducible and testable (NFR-03 / NFR-04).

To explore a single stage interactively, import the package directly — each
stage is a small, independent function:

```python
import sys; sys.path.insert(0, "..")
from scholargrid.config import load_config
from scholargrid.data_ingest import build_corpus
from scholargrid.embeddings import Embedder, paper_text
from scholargrid.reduce import reduce_analytical, project_2d
from scholargrid.clustering import cluster
from scholargrid.labeling import generate_labels
from scholargrid.growth import compute_growth
from scholargrid.sparse import detect_sparse

cfg = load_config("../configs/config.yaml")
df  = build_corpus(cfg)
emb = Embedder(cfg.resolve_embedding_backend(), cfg)
X   = emb.fit_transform(paper_text(df))
# ... etc. (mirrors pipeline/run_pipeline.py)
```

Suggested notebooks to add as the project grows:

| Notebook | Stage | Requirement |
|---|---|---|
| `01_data_ingest.ipynb` | corpus build & cleaning | FR-01 |
| `02_embeddings_clustering.ipynb` | embeddings → UMAP → HDBSCAN | FR-02, FR-03 |
| `03_labels_landscape.ipynb` | c-TF-IDF labels, 2D map | FR-04, FR-05 |
| `04_growth_sparse.ipynb` | growth & sparse neighbourhoods | FR-07, FR-08 |
| `05_validation.ipynb` | validation & robustness | FR-15–FR-18 |

Prototype in a notebook, then fold stable logic back into `scholargrid/` so the
pipeline remains the single reproducible source of truth.
