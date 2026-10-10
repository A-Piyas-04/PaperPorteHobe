# 🔭 ScholarGrid — Research Landscape Explorer

> Turn arXiv paper discovery from keyword hunting into **landscape understanding**.
> Search → landscape → trends → evidence → investigation leads.

ScholarGrid transforms a corpus of recent arXiv **computer-science** papers into
an interactive semantic research landscape. Users can understand major research
areas, search semantically, observe publication-growth patterns, surface sparse
("investigation lead") neighbourhoods, and inspect the papers behind every
signal.

It is an **exploration and decision-support tool** — *not* an automatic
research-gap or thesis-topic generator. See [`Docs/ScholarGrid_SRS.md`](Docs/ScholarGrid_SRS.md)
for the full requirements this implements.

This repository contains the **version 2 pipeline** and a guided research UI.
The original SRS records the analytical scope; the [UX review](Docs/ux-review.md)
documents the current interface and remaining readiness limits.

---

## Architecture

The expensive analysis runs **offline** and writes precomputed artifacts; the
deployed app stays lightweight, doing only live query embedding + nearest-
neighbour retrieval (NFR-01, Deployment Requirements).

```
arXiv metadata
   → clean/filter (cs.*, window, dedup)        FR-01
   → title+abstract embeddings                 FR-02
   → UMAP reduce → HDBSCAN clustering          FR-03
   → c-TF-IDF labels + representative papers    FR-04
   → separate 2D projection (landscape)        FR-05
   → normalised publication growth             FR-07
   → sparse-neighbourhood leads                FR-08
   → validation & robustness                   FR-15..18
   → exported artifacts  ──►  Streamlit app     FR-05/06/10/12/13/14
```

### Backends (graceful degradation)

The pipeline uses the SRS baseline methods when their libraries are installed,
and reproducible scikit-learn fallbacks otherwise. The resolved choice is
recorded in `data/processed/meta.json` and shown in the app's Methodology tab.

| Stage | Preferred (SRS) | Fallback |
|---|---|---|
| Embeddings (FR-02) | `sentence-transformers` all-MiniLM-L6-v2 | TF-IDF → TruncatedSVD → L2 |
| Reduction (FR-03/05) | UMAP | PCA |
| Clustering (FR-03) | HDBSCAN | DBSCAN (k-distance `eps`, noise preserved) |

---

## Repository layout (NFR-04)

```
configs/      config.yaml (all parameters & seeds) + validation_queries.json
scholargrid/  reusable pipeline library (one module per stage)
pipeline/     run_pipeline.py — offline orchestrator
app/          streamlit_app.py — the web application
notebooks/    interactive exploration (see notebooks/README.md)
data/raw/     harvested corpus (gitignored)
data/processed/ precomputed artifacts consumed by the app (gitignored)
reports/      generated validation report (md + json)
Docs/         proposal, SRS, implementation plans
```

---

## Quickstart

```bash
# 1. Install (core stack; add the optional baseline backends if you want them)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional SRS baseline methods:
# pip install sentence-transformers umap-learn hdbscan

# 2. Build the artifacts (harvests arXiv by default; see configs/config.yaml)
python pipeline/run_pipeline.py          # or skip and click "Load demo data" in the app

# 3. Launch the app
streamlit run app/streamlit_app.py       # app.py also works (Hugging Face Spaces entry)
```

New here? See [`Docs/next-instructions.md`](Docs/next-instructions.md) for step-by-step
setup and deployment, and [`Docs/README.md`](Docs/README.md)
for the technical reference.

### Choosing a data source

Edit `data.source` in [`configs/config.yaml`](configs/config.yaml):

- `arxiv_api` *(dev default)* — live harvest from the arXiv search API.
- `kaggle` — streams a downloaded `arxiv-metadata-oai-snapshot.json`
  (`data.kaggle_json`) into the Parquet/DuckDB paper store.
- `oai_pmh` — incremental, resumable harvest from arXiv OAI-PMH (weekly refresh).
- `synthetic` — deterministic offline demo corpus (no network), for CI and demos.

Falling back to `synthetic` when a harvest fails is allowed only when
`data.allow_synthetic_fallback: true` (dev). Production configs refuse it, and
the app shows a banner whenever synthetic data is being served.

### Configs and corpus size

| Config | Purpose |
|---|---|
| `configs/config.yaml` | Local development (small sample, permissive gates) |
| `configs/production.yaml` | Production build from the Kaggle snapshot (`profile: standard`, strict gates) |
| `configs/refresh.yaml` | Weekly OAI-PMH refresh on top of production |
| `configs/ci.yaml` | Offline nightly regression run |

`data.profile` picks the corpus size: `dev` (5k), `standard` (50k), `full`
(no cap) or `custom` (`max_papers`). Secrets (`OPENALEX_MAILTO`, `SENTRY_DSN`,
`S2_API_KEY`) are read from environment variables only.

### Pipeline, releases and operations

```bash
python pipeline/run_pipeline.py --config configs/production.yaml   # full build + publish
python pipeline/run_pipeline.py --from-stage analyze               # rerun from a stage
python -m scholargrid.release list | rollback <release-id>
```

Each run writes a staging bundle, validates it, and publishes a versioned
release (`data/releases/<id>/` with `manifest.json` and `changelog.json`)
only if the required validation gates pass. See
[`Docs/runbook.md`](Docs/runbook.md) for rebuilds, rollbacks, refresh failures
and secrets.

### Development

```bash
pip install -e ".[ml,app,dev]"
pytest                 # unit, integration and Streamlit smoke tests
ruff check . && mypy
pre-commit install
docker build -t scholargrid .   # production image (non-root, health-checked)
```

---

## The application (SRS §16.1)

The application follows **Find papers - understand the area - build a reading list**.

- **Find papers:** describe an interest, use example searches, and inspect ranked results.
- **Quick tour:** four short steps, with Back, Skip, and replay on the home page.
- **Workspace:** browse readable area cards, then select papers in a two-column reading view. On phones, switch between the list and details. Selection changes only on click.
- **Related papers:** follow connections for one selected paper, without a full-corpus network.
- **Reading list:** save search results for the current session and export CSV or BibTeX.
- **More:** area summaries, trends, investigation leads, methodology, and dataset provenance.
- **Trust:** original arXiv links, coverage disclosures, demo/degraded-search notices, and existing reliability gates.

See the [documentation guide](Docs/README.md) and [UX review](Docs/ux-review.md) for current behavior, historical-document corrections, and remaining limits. Session data is not durable storage. This local UI redesign does not establish production launch readiness.

---

## Reproducibility (NFR-03)

Pinned dependencies, a single config with seeds and parameters, stored dataset
snapshot info, saved embeddings / cluster assignments / 2D coordinates, and a
`meta.json` associating every run with its pipeline version and backend choices.

## Validation (FR-15–FR-18)

`python pipeline/run_pipeline.py` also writes
[`reports/validation_report.md`](reports/validation_report.md) and appends to
`reports/validation_history.jsonl`:

- Search: nDCG@10, MRR, precision@10 and recall@25 over 129 queries, for the
  hybrid, dense-only and BM25-only modes.
- Clusters: DBCV, NPMI coherence, keyword diversity, and subsample stability
  (ARI and per-cluster survival).
- Growth: minimum-count gating, bootstrap confidence intervals, stability across
  windows and cutoffs, and a 12-month backtest.
- Data quality checks, ANN recall, OpenAlex match rates, and a gates table that
  decides whether the run is published.

---

## Scope of version 1

**Implemented** — all *Must* requirements, plus *Should*: sparse-neighbourhood
detection, category/date filtering, and the evidence UI.

**Version 2 (production readiness)** adds a scalable data layer (Kaggle
streaming, OAI-PMH, Parquet/DuckDB), richer OpenAlex enrichment with citation
history, incremental embeddings, an ANN index, hybrid BM25 + dense search,
honest growth statistics, versioned releases with rollback, a weekly refresh
workflow, CI and Docker. See
[`Docs/production-readiness-plan.md`](Docs/production-readiness-plan.md).

**Not yet implemented** (SRS stretch items): semantic-citation missing-link
analysis (FR-09) and beginner research-entry recommendations.

## What ScholarGrid does *not* claim (SRS §21)

It does not decide whether a topic is valuable, novel, publishable, or suitable
for a given lab; it does not equate publication volume with quality; and it does
not treat embedding proximity as scientific compatibility. It surfaces
**evidence-backed areas worth further literature investigation** and always lets
you inspect the underlying papers.
