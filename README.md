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

This repository contains **version 1** — all SRS *Must* requirements plus the
*Should* items (sparse detection, category/date filtering, evidence UI).

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

New here? See [`next-instructions.md`](next-instructions.md) for step-by-step
setup and deployment, and [`implementation-details.md`](implementation-details.md)
for the technical reference.

### Choosing a data source

Edit `data.source` in [`configs/config.yaml`](configs/config.yaml):

- `arxiv_api` *(default)* — live harvest from the arXiv API.
- `kaggle` — point `data.kaggle_json` at a downloaded
  `arxiv-metadata-oai-snapshot.json`.
- `synthetic` — deterministic offline demo corpus (no network), useful for CI
  and for trying the app without a harvest.

Network sources fall back to `synthetic` automatically if the harvest fails, so
a run always completes.

---

## The application (SRS §16.1)

A clean, low-text GUI (no neon/glassmorphism) built around a single interactive
map and one adaptive side panel.

- **Research landscape** — interactive 2D WebGL map; **click or lasso points to
  inspect papers**, focus an area to dim the rest.
- **Semantic search bar** — one box + example chips; query embedded with the
  same model; matches starred on the map, cluster distribution shown, ambiguity
  flagged.
- **Adaptive panel** — search results → selected papers → cluster details
  (label, keywords, representative papers, monthly counts, relative growth) →
  *Research areas* overview.
- **Growth / Investigation leads / How it works / Limitations / Dataset** —
  collapsed expanders (FR-13/FR-14), kept out of the way until needed.
- **Evidence everywhere** — every item links back to real arXiv papers (NFR-06).
- **First run** — one-click *Load demo data* (offline) or the real-harvest
  command.

---

## Reproducibility (NFR-03)

Pinned dependencies, a single config with seeds and parameters, stored dataset
snapshot info, saved embeddings / cluster assignments / 2D coordinates, and a
`meta.json` associating every run with its pipeline version and backend choices.

## Validation (FR-15–FR-18)

`python pipeline/run_pipeline.py` also writes
[`reports/validation_report.md`](reports/validation_report.md):
cluster cohesion & silhouette, semantic-search precision@k over a 16-query
known-item benchmark, growth stability across 3/6/12-month windows and alternate
cutoffs, and sparse-lead projection robustness.

---

## Scope of version 1

**Implemented** — all *Must* requirements, plus *Should*: sparse-neighbourhood
detection, category/date filtering, and the evidence UI.

**Not yet implemented** (SRS stretch items): semantic-citation missing-link
analysis via OpenAlex (FR-09), monthly refresh pipelines, and beginner
research-entry recommendations.

## What ScholarGrid does *not* claim (SRS §21)

It does not decide whether a topic is valuable, novel, publishable, or suitable
for a given lab; it does not equate publication volume with quality; and it does
not treat embedding proximity as scientific compatibility. It surfaces
**evidence-backed areas worth further literature investigation** and always lets
you inspect the underlying papers.
