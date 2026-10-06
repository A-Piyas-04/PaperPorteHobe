# ScholarGrid — Production Readiness Plan

How to take ScholarGrid from a working MVP (v1) to a usable, trustworthy,
production-grade research landscape tool (v2).

> **Status on 2026-10-06:** v1 runs end to end on real arXiv data. It works well
> for **semantic search, clustering, paper browsing and the landscape UI**, but
> the corpus is far too small and too narrow in time for **trends, research-gap
> signals or strategic decisions**.

---

## 0. Where we are today (baseline)

| Item | Current value | Verdict |
|---|---|---|
| Source | arXiv API (`data.source: arxiv_api`), cached in `data/raw/arxiv_harvest.csv` | OK for a demo |
| Papers | 1,275 | Far too few |
| Actual time span | 2026-09-25 → 2026-10-02 (**8 days**) | Far too short |
| Configured window | 2023-01-01 → 2026-10-05 | Not reached: the harvester takes the *newest* N papers per query |
| Categories | 8 `cs.*` queries (LG, CL, CV, CR, RO, AI, DC, SE) | Missing most of CS |
| Backends | MiniLM + UMAP + HDBSCAN | Good baseline |
| Clusters | 26, noise 38.7% | Reasonable |
| Search precision@k (16 queries) | 0.88 | Decent but the benchmark is small and lenient |
| Silhouette | 0.51 | Decent |
| Growth stability | **26 / 26 clusters unstable**, `sufficient_history: false` | Not usable |
| Citations / references | 0 / 0 (papers are a week old) | Not usable yet |
| Tests | none | Blocker for production |

### What v1 can honestly claim
"Here are recent arXiv CS papers, grouped semantically, with search and a
similarity graph."

### What v1 must not claim yet
"This is the real research landscape", "this area is growing", "this is a
research gap".

---

## 1. Goals and non-goals

### Goals
1. **Data that represents the field**: several years of CS papers, broad
   category coverage, refreshed on a schedule.
2. **Trustworthy signals**: growth and sparse-neighbourhood leads are shown only
   when there is enough history and they pass validation.
3. **Citation awareness**: citation counts, references and venues that become
   meaningful as papers age.
4. **Production engineering**: tests, CI, monitoring, versioned data releases,
   reproducible builds, safe failures.
5. **Scales** to 50k–500k papers without the app becoming slow.

### Non-goals (still out of scope)
- Declaring topics novel, valuable, or publishable (SRS §21 stays in force).
- User accounts / paid features in the first production release.
- Full-text (PDF) analysis. Title + abstract stay the unit of analysis.

---

## 2. Guiding rules (carry these through every phase)

These are the lessons from v1 and they are not optional:

1. **Several years of papers, not one week.** Target at least 3 years
   (2023-01 onward); 5 years is better for trend baselines.
2. **Much larger corpus.** Start at 8k–50k papers, design for 500k+.
3. **Use a bulk source for history.** The arXiv search API is for "latest
   papers", not history. Use the Kaggle arXiv snapshot or arXiv OAI-PMH for the
   backfill, and OpenAlex / Semantic Scholar for metadata and citations.
4. **Refresh citations as papers age.** Citations for week-old papers are
   ~0; re-enrich on a schedule and store citation history.
5. **Gate trends on history.** Show growth only when
   `growth.sufficient_history` is true *and* the cluster passes stability
   validation. Otherwise show counts only, clearly labelled.
6. **Never silently fake data.** A failed harvest in production must fail the
   build, not fall back to `synthetic`.

---

## 3. Phased roadmap

| Phase | Theme | Rough effort | Outcome |
|---|---|---|---|
| 1 | Data foundation | 1–2 weeks | Multi-year, broad, clean corpus |
| 2 | Enrichment (citations, venues, authors) | 1 week | Citation-aware papers |
| 3 | Scalable pipeline | 1–2 weeks | 50k–500k papers in reasonable time |
| 4 | Analytical quality & validation | 1–2 weeks | Trends and leads that can be trusted |
| 5 | App: performance & UX for scale | 1–2 weeks | Fast app at large corpus sizes |
| 6 | Engineering quality (tests, CI, code health) | 1 week, then ongoing | Safe to change |
| 7 | Deployment, operations, monitoring | 1 week | Reliable public service |
| 8 | Refresh automation & data releases | 3–5 days | Always current, versioned |
| 9 | Security, legal, compliance | 2–3 days | Safe and properly licensed |
| 10 | Launch readiness | 2–3 days | Go-live checklist passed |

Phases 1–4 are the critical path. Phase 6 should start in parallel with
Phase 1 (write tests as each module is touched).

---

## Phase 1 — Data foundation

**Goal:** a corpus that actually covers the CS research landscape over time.

### 1.1 Pick the historical source

| Option | Pros | Cons | Use for |
|---|---|---|---|
| **Kaggle arXiv snapshot** (`arxiv-metadata-oai-snapshot.json`, ~4 GB, 2.5M+ papers) | Full history, one download, has categories and versions | Updated roughly weekly by Kaggle, large file | **Initial backfill** (recommended) |
| **arXiv OAI-PMH** (`export.arxiv.org/oai2`, `metadataPrefix=arXiv`) | Official bulk interface, incremental by date (`from=`/`until=`), resumable | Slower, must respect rate limits | **Incremental daily/weekly updates** |
| arXiv search API (current) | Simple | Pagination limits, sorted newest first, not for bulk | Ad hoc only |
| OpenAlex works API | Rich metadata, citations, topics, institutions | arXiv matching needs DOI lookup | Enrichment (Phase 2) |
| Semantic Scholar API | Citations, influential citations, SPECTER embeddings | API key and rate limits | Enrichment / optional embeddings |

**Decision:** Kaggle snapshot for backfill + OAI-PMH for incremental updates +
OpenAlex for enrichment. Keep the arXiv API source only for quick local demos.

### 1.2 Fix the ingestion code (`scholargrid/data_ingest.py`)

Known issues found in v1:

- [ ] **Kaggle loader reads the oldest papers.** `_load_kaggle` stops after
  `max_papers * 4` lines *from the start of the file*. The snapshot is roughly
  ordered by arXiv ID (oldest first), so with `date_start: 2023-01-01` almost
  everything read gets filtered out. Fix: stream the whole file, filter by
  category and date while streaming, then sample.
- [ ] **arXiv API harvest only gets the newest N papers per query**, which is
  why the "2023–2026" window became 8 days. Replace with OAI-PMH for
  date-ranged harvesting.
- [ ] **Silent synthetic fallback.** If the harvest fails, `build_corpus`
  falls back to synthetic data. Add `data.allow_synthetic_fallback`
  (default `false` in production) and fail loudly instead.
- [ ] **Date semantics.** Store both `first_submitted` (v1 date) and
  `last_updated`. Use `first_submitted` for growth.
- [ ] **Category handling.** `categories.str.contains("cs.")` also matches
  things like `physics.cs...` edge cases; split on whitespace and check the
  prefix per category. Keep `primary_category` from the source, not "first
  listed".
- [ ] **De-duplication.** Dedupe by base arXiv ID (strip version), then by
  normalised title (lowercase, punctuation stripped). Keep the latest version's
  metadata.
- [ ] **Text normalisation.** Strip LaTeX markup (`$...$`, `\emph{}`),
  normalise unicode, collapse whitespace.
- [ ] **Language filter.** Drop non-English abstracts (cheap detector such as
  `langdetect` or a fastText LID model).

### 1.3 Storage: move from CSV to a real data layer

CSV + loading everything into pandas will not scale or support incremental
updates.

- [ ] **Raw layer:** Parquet files partitioned by `year_month`
  (`data/raw/arxiv/year_month=2025-03/part-*.parquet`).
- [ ] **Working store:** DuckDB (single file, zero ops, fast analytics) for v2.
  Move to PostgreSQL + pgvector only if multi-user writes or a separate API
  service become necessary.
- [ ] **Schema** (one row per paper):
  `arxiv_id (PK), version, title, abstract, authors (list), categories (list),
  primary_category, first_submitted, last_updated, doi, journal_ref,
  license, ingested_at, source`.
- [ ] Keep `papers.csv` export only as a convenience artefact, not as the
  system of record.

### 1.4 Corpus scope and sizing

- [ ] **Time window:** 2021-01-01 → today (5 years) for trend baselines;
  minimum 2023-01-01.
- [ ] **Categories:** all `cs.*` primaries (~40 categories), with
  cross-listed papers included via `categories`.
- [ ] **Size tiers** (configurable via `data.profile`):

| Profile | Papers | Use |
|---|---|---|
| `dev` | ~5k | Local development, CI |
| `standard` | ~50k (stratified sample) | First production release |
| `full` | all CS papers in window (~300k–500k) | Final target |

- [ ] **Stratified sampling** for `standard`: sample per
  `(primary_category, year_month)` so small categories and older months are not
  drowned out; store sampling weights so growth numbers can be reweighted.

### 1.5 Data quality checks (run on every build)

Fail the build if any check fails:

- [ ] Time span ≥ configured minimum (e.g. 24 months).
- [ ] Every month in the window has ≥ N papers (no silent gaps).
- [ ] No category above X% of the corpus unless expected.
- [ ] Missing title/abstract rate < 0.5%; duplicate rate after dedup = 0.
- [ ] Row count within ±20% of the previous release (catches broken harvests).

Use plain assertions or `pandera` / Great Expectations; write results into the
data release manifest (Phase 8).

### Phase 1 acceptance
- Corpus spans ≥ 3 years, ≥ 50k papers (`standard` profile), all `cs.*`.
- `meta.json → snapshot` shows the real span; quality checks pass.
- No synthetic data reachable in production config.

---

## Phase 2 — Enrichment: citations, references, venues, authors

**Goal:** citation-aware papers whose numbers improve as papers age.

### 2.1 OpenAlex (extend `scholargrid/enrich.py`)
- [ ] Set `enrich.mailto` (polite pool, higher rate limits).
- [ ] Use the bulk/`cursor` pagination for large batches; optionally use the
  OpenAlex snapshot (S3) for the initial backfill instead of the API.
- [ ] Store more fields: `cited_by_count`, `counts_by_year`,
  `referenced_works`, `primary_location.source` (venue), `topics`,
  `authorships` (author IDs, institutions), `publication_date`, `type`.
- [ ] Match on arXiv DOI first (`10.48550/arxiv.<id>`), then fall back to
  title + first author fuzzy match for older papers without DataCite DOIs.
- [ ] Track match rate per month and category; alert if it drops.

### 2.2 Citation refresh schedule
Citations need time to accumulate:

| Paper age | Refresh interval |
|---|---|
| < 3 months | weekly |
| 3–24 months | monthly |
| > 24 months | quarterly |

- [ ] Store **citation history** (`paper_id, snapshot_date, cited_by_count`)
  so the app can show citation velocity, not just a total.
- [ ] Replace the single JSON cache (`data/raw/openalex.json`) with a table in
  the data store.

### 2.3 Semantic Scholar (optional, second source)
- [ ] Get an API key; pull `citationCount`, `influentialCitationCount`, and
  optionally SPECTER2 embeddings for comparison.
- [ ] Cross-check citation counts with OpenAlex; log large disagreements.

### 2.4 Author and venue normalisation
- [ ] Use OpenAlex author IDs to merge name variants.
- [ ] Map venues to a canonical list (conference/journal), keep `arXiv-only`
  explicitly.

### 2.5 Using citations in the product
- [ ] Citation-aware ranking option in search (similarity × log citations,
  age-normalised).
- [ ] Bibliographic coupling and co-citation edges in the graph
  (`scholargrid/graph.py` already supports a coupling boost).
- [ ] FR-09 (semantic-close but citation-sparse "potential missing
  connections") becomes feasible once reference coverage is > 60%.

### Phase 2 acceptance
- OpenAlex match rate ≥ 85% for papers older than 3 months.
- Reference coverage ≥ 60% for papers older than 6 months.
- Citation history recorded across at least two refreshes.

---

## Phase 3 — Scalable pipeline

**Goal:** the offline pipeline handles 50k–500k papers reliably and
incrementally.

### 3.1 Embeddings
- [ ] **Model choice:** benchmark `all-MiniLM-L6-v2` (current, fast) vs
  scientific models: `allenai/specter2` (with proximity adapter),
  `BAAI/bge-small-en-v1.5`, `nomic-embed-text-v1.5`. Pick by the expanded
  search benchmark (Phase 4) and cost.
- [ ] **Incremental embedding:** embed only new/changed papers; store vectors
  keyed by `arxiv_id + text hash + model version`.
- [ ] **GPU optional:** support CUDA when available; batch size from config.
- [ ] **Storage:** float16 vectors on disk (halves size, negligible quality
  loss); keep model name/version in metadata.
- [ ] Pin the model revision (Hugging Face commit hash) so query embeddings in
  the app always match corpus embeddings.

### 3.2 Nearest-neighbour search
`cosine_topk` does brute force over the full matrix; fine at 1k, too slow and
memory-heavy at 500k.
- [ ] Add an ANN index: **FAISS** (`IndexHNSWFlat` or `IVF+PQ`) or
  **hnswlib**. Build offline, ship the index in the bundle.
- [ ] Keep brute force as the exact fallback for small corpora and for
  validating ANN recall (target recall@25 ≥ 0.98).
- [ ] Use the same index for the graph's k-nearest-neighbour edges.

### 3.3 Reduction and clustering at scale
- [ ] UMAP on 500k × 384 is feasible but slow; options: fit UMAP on a
  stratified sample (50k–100k) and `transform` the rest, or use cuML on GPU.
- [ ] HDBSCAN: tune `min_cluster_size` relative to corpus size (e.g. 0.05–0.1%
  of papers) and keep `leaf` selection; consider a **two-level hierarchy**
  (broad areas → subtopics) so the UI is navigable with hundreds of clusters.
- [ ] **Cluster identity across runs:** match new clusters to previous ones
  (centroid similarity + member overlap / Jaccard) so cluster IDs and labels
  stay stable between refreshes. Required for meaningful trend history.

### 3.4 Pipeline orchestration
`runner.run` executes everything in one process every time.
- [ ] Split into idempotent stages with cached outputs: `ingest → clean →
  enrich → embed → index → reduce → cluster → label → project → growth →
  sparse → graph → validate → publish`.
- [ ] Each stage reads/writes versioned artefacts and can be re-run alone
  (`python -m scholargrid.pipeline --stage embed`).
- [ ] Orchestrator: start with a simple Makefile / `invoke` / Python CLI with
  stage caching; adopt Prefect or Dagster only if scheduling and retries
  outgrow GitHub Actions.
- [ ] Structured logging (JSON) with stage timings and row counts.
- [ ] Memory: process in chunks; avoid holding multiple full copies of the
  DataFrame.

### Phase 3 acceptance
- `standard` profile (50k) builds end to end in < 1 hour on a single machine.
- Incremental weekly update (new papers only) runs in < 15 minutes.
- ANN recall@25 ≥ 0.98 vs exact search.
- ≥ 80% of clusters keep their ID between two consecutive refreshes.

---

## Phase 4 — Analytical quality and validation

**Goal:** every signal shown in the app is validated, and anything that is not
validated is hidden or clearly labelled.

### 4.1 Search quality
The current benchmark is 16 queries judged by a lenient rule (category *or*
title keyword match).
- [ ] Expand to **100–200 queries** covering all major CS areas, including
  hard/ambiguous ones.
- [ ] Build **graded relevance judgments** (0/1/2) for top-20 results, by
  hand or with an LLM judge + human spot checks (≥ 20% audited).
- [ ] Report **nDCG@10, MRR, recall@25** in addition to precision@k.
- [ ] Add hybrid retrieval: BM25 (e.g. `rank_bm25` or DuckDB FTS) + dense,
  fused with Reciprocal Rank Fusion. Exact-term queries (model names,
  acronyms) improve a lot with this.
- [ ] Optional cross-encoder re-ranker on the top 50 (e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) if latency allows.
- [ ] Targets: nDCG@10 ≥ 0.75, precision@10 ≥ 0.85 on the expanded set.

### 4.2 Cluster quality
- [ ] Metrics: silhouette, DBCV (density-based), noise fraction, topic
  coherence (NPMI on keywords), keyword diversity.
- [ ] **Human review:** for each of the top 30 clusters, a reviewer rates label
  quality and coherence of 10 sampled papers. Keep a review log.
- [ ] Better labels: keep c-TF-IDF keywords but add an optional LLM-generated
  short name + one-line description, constrained to the keywords and
  representative titles, with the keywords always shown alongside.
- [ ] Noise handling: target noise ≤ 30%; offer "nearest area" assignment for
  noise papers in the UI, clearly marked as soft.
- [ ] Stability: re-run clustering under 3–5 seeds / subsamples, report
  Adjusted Rand Index; flag clusters that do not survive.

### 4.3 Growth / trends (the biggest v1 gap)
- [ ] **Hard gate:** show growth for a window only if
  `growth.sufficient_history` is true for that window *and* the cluster has
  ≥ 30 papers in each compared period.
- [ ] Unstable clusters (CV > 0.5 across windows and cutoffs) show counts only,
  with a "trend not reliable" badge.
- [ ] Correct for arXiv's overall growth (already normalised by the corpus) and
  for sampling weights when using the `standard` profile.
- [ ] Add **uncertainty**: bootstrap confidence intervals on relative growth;
  only call something "growing" if the lower bound > 1.
- [ ] Monthly time series with seasonality awareness (conference deadline
  spikes); consider a simple trend fit (e.g. Poisson regression on monthly
  counts) instead of only two-window ratios.
- [ ] **Backtest:** compute growth as of 12 months ago and check whether
  "growing" clusters actually kept growing; report hit rate.
- [ ] Remove the `+1` smoothing distortion for small counts or require minimum
  counts so tiny clusters cannot show huge ratios.

### 4.4 Sparse-neighbourhood leads
- [ ] Only run when the corpus is ≥ `standard` size (sparsity in a 1k sample is
  mostly sampling noise).
- [ ] Increase robustness checks: 5+ seeds, 5+ bootstrap subsamples, and an
  alternative embedding model; require survival in ≥ 70%.
- [ ] Control for sampling: a region is only "sparse" if it stays sparse in the
  `full` corpus, not just the sample.
- [ ] Attach evidence: nearby papers, their dates and citations, and the
  nearest clusters' growth.
- [ ] Keep wording: "investigation lead", never "research gap".
- [ ] Expert review of a sample of leads each release; track precision.

### 4.5 Validation as a release gate
- [ ] `validate()` returns pass/fail per check with thresholds from config.
- [ ] The publish stage refuses to publish a bundle that fails required checks
  (search nDCG, quality checks, cluster stability), and writes a clear report.
- [ ] Keep `reports/validation_report.{md,json}` per release and a trend of
  metrics across releases.

### Phase 4 acceptance
- Expanded benchmark targets met.
- Growth shown only for clusters passing the gate; backtest hit rate reported.
- Leads reviewed by a human; precision ≥ 60% judged "worth investigating".

---

## Phase 5 — App: performance and UX for scale

**Goal:** the Streamlit app stays fast and clear with 50k–500k papers.

### 5.1 Loading and memory
- [ ] Load embeddings with `np.load(..., mmap_mode="r")`; load the ANN index
  instead of the full matrix for search.
- [ ] Load paper metadata from DuckDB/Parquet on demand (selected columns,
  filtered rows) instead of the full `papers.csv`.
- [ ] Cache with `st.cache_resource` (index, model) and `st.cache_data`
  (query results keyed by query + filters).
- [ ] Load the embedding model once per process; pre-warm on startup.

### 5.2 Explore graph at scale
`edges.json` with ~6k edges is fine; 500k papers × 6 neighbours is 3M edges.
- [ ] Level of detail: show cluster-level nodes when zoomed out, papers when
  zoomed into an area.
- [ ] Cap rendered nodes (e.g. ≤ 20k) by filters, area, or sampling; stream
  the rest on demand.
- [ ] Ship graph data in a compact binary/columnar format instead of JSON.

### 5.3 Search UX
- [ ] Filters: date range, categories, citation count, venue, has-code.
- [ ] Sorting: relevance, newest, most cited, citation velocity.
- [ ] "More like this" from any paper; export results (CSV/BibTeX).
- [ ] Show why a result matched (highlight terms for the BM25 part).

### 5.4 Honest signals in the UI
- [ ] Every growth number shows the window, absolute counts, confidence
  interval, and a reliability badge.
- [ ] Dataset panel shows real span, paper count, last refresh, citation
  coverage, and the data release version.
- [ ] Banner when data is stale (> 2 refresh intervals old).

### 5.5 Accessibility and polish
- [ ] Keyboard navigation, colour-blind-safe palette, sufficient contrast.
- [ ] Mobile-friendly fallback (list views instead of graph).
- [ ] Empty/error states for every view; no raw tracebacks shown to users.

### Phase 5 acceptance
- Cold start < 15 s; search p95 latency < 500 ms at 50k papers (< 1 s at 500k).
- Memory usage < 2 GB at 50k papers.
- Explore view stays interactive (> 30 fps) with filters applied.

---

## Phase 6 — Engineering quality

**Goal:** the codebase is safe to change. There are currently **no tests**.

### 6.1 Tests (`tests/`, `pytest`)
- [ ] **Unit tests** per module: cleaning rules, dedup, date parsing, growth
  formula and gating, c-TF-IDF labels, cosine top-k, cluster ID matching,
  enrichment parsing (with recorded API fixtures).
- [ ] **Integration test:** run the full pipeline on the synthetic corpus
  (`dev` size) and assert bundle shape and validation output.
- [ ] **Regression tests:** golden metrics on a frozen small real sample
  (precision@k, cluster count range) so model/library upgrades are caught.
- [ ] **App smoke test:** Streamlit `AppTest` loads each page without errors.
- [ ] Coverage target ≥ 80% for `scholargrid/`.

### 6.2 Code health
- [ ] Lint/format: `ruff` (lint + format); type check: `mypy` (or `pyright`)
  on `scholargrid/`.
- [ ] `pre-commit` hooks for ruff, mypy, trailing whitespace, large files.
- [ ] Packaging: `pyproject.toml` with optional extras
  (`[ml]`, `[dev]`, `[app]`), replacing commented-out lines in
  `requirements.txt`.
- [ ] Lock dependencies (`uv lock` or `pip-tools`), including
  `sentence-transformers`, `umap-learn`, `hdbscan`, `faiss-cpu`.
- [ ] Config validation with `pydantic` so bad YAML fails fast with a clear
  message.
- [ ] Replace pickles in the bundle (`embedder/*.pkl`) with safer formats
  (JSON metadata + `joblib`/`safetensors`), and never load bundles from
  untrusted sources.

### 6.3 CI (GitHub Actions)
- [ ] On every PR: lint, type check, unit tests, integration test on synthetic
  data, frontend build (`npm ci && npm run build`).
- [ ] Nightly: regression tests on the frozen real sample.
- [ ] Block merge on failures; require review.

### Phase 6 acceptance
- CI green on `main`; coverage ≥ 80%; no lint/type errors.

---

## Phase 7 — Deployment, operations, monitoring

**Goal:** a reliable public service.

### 7.1 Hosting options

| Option | Fit | Notes |
|---|---|---|
| Hugging Face Spaces (current plan) | Good for `dev`/small `standard` | Free CPU tier has limited RAM; persistent storage is paid; fine for demo |
| Hugging Face Spaces (upgraded hardware + persistent storage) | Good for `standard` | Simple, keeps current workflow |
| Docker on a VM / Fly.io / Render / Railway | Good for `standard`/`full` | More control, predictable cost |
| Cloud Run / ECS + object storage | Good for `full` | Autoscaling, more ops work |

**Recommendation:** containerise first (Dockerfile), deploy to HF Spaces with
persistent storage for the first production release, keep the image portable
for a move to a VM/Cloud Run later.

### 7.2 Container and artefacts
- [ ] `Dockerfile` (multi-stage, slim Python 3.12, non-root user, model weights
  baked in or cached on a volume).
- [ ] Artefacts (bundle, index) stored in object storage or a Hugging Face
  Dataset repo, versioned; the app downloads the pinned release at startup.
- [ ] Stop pushing data through Git LFS on a deploy branch once the dataset
  repo is in place.

### 7.3 Monitoring
- [ ] Health endpoint/check page; uptime monitor (e.g. UptimeRobot, Better
  Stack).
- [ ] Error tracking: Sentry for app and pipeline.
- [ ] Metrics: search latency, error rate, memory, active sessions.
- [ ] Privacy-friendly product analytics (popular queries, zero-result
  queries) to improve the benchmark; no personal data stored.

### 7.4 Reliability
- [ ] Graceful degradation: if the model fails to load, fall back to BM25
  search with a visible notice.
- [ ] Rollback: the app can be pointed at the previous data release with one
  config change.
- [ ] Runbook (`Docs/runbook.md`): how to rebuild, roll back, rotate keys,
  handle API outages.

### Phase 7 acceptance
- Public URL up ≥ 99% over a month; errors reported to Sentry; rollback tested.

---

## Phase 8 — Refresh automation and data releases

**Goal:** the data stays current without manual work, and every release is
traceable.

- [ ] **Weekly job** (GitHub Actions scheduled workflow or a small worker):
  OAI-PMH incremental harvest → clean → enrich new papers → embed new papers →
  update index → reassign clusters (`approximate_predict` for HDBSCAN) →
  recompute growth → validate → publish.
- [ ] **Monthly/quarterly job:** full re-cluster and re-projection with cluster
  ID matching to the previous release; citation refresh per the schedule in
  Phase 2.
- [ ] **Data release manifest** per build: release version, corpus span, paper
  count, source snapshot dates, model + library versions, config hash,
  validation results, quality-check results.
- [ ] **Release versioning:** `YYYY.MM.DD` data release IDs; keep the last N
  releases.
- [ ] **Changelog for users:** "N new papers, M new areas, areas that merged or
  split".
- [ ] Notifications (email/Slack/GitHub issue) on failed refreshes or failed
  validation.

### Phase 8 acceptance
- Four consecutive automated weekly refreshes succeed without manual steps.
- Every published release has a manifest and a validation report.

---

## Phase 9 — Security, legal, compliance

- [ ] **Licences:** arXiv metadata is CC0 for metadata, but abstracts carry the
  paper's licence; show abstracts with attribution and link to arXiv, do not
  redistribute PDFs. Follow arXiv API Terms of Use and the arXiv brand
  guidelines ("Thank you to arXiv for use of its open access interoperability").
- [ ] OpenAlex data is CC0; Semantic Scholar has its own licence (check terms
  before displaying data).
- [ ] Rate limits and polite headers (`mailto`, `User-Agent`) for every API.
- [ ] Secrets (API keys) in environment variables / platform secrets, never in
  `config.yaml` or Git.
- [ ] Dependency scanning (Dependabot, `pip-audit`); container scanning.
- [ ] Input handling: limit query length, sanitise any HTML rendered with
  `unsafe_allow_html=True` (titles/abstracts from external sources must be
  escaped).
- [ ] Privacy policy and terms page if analytics are collected.
- [ ] Clear disclaimer in the app (already in About): exploration tool, not a
  judgment of research value.

---

## Phase 10 — Launch readiness checklist

Go live only when every box is ticked:

**Data**
- [ ] Corpus ≥ 3 years and ≥ 50k papers, all `cs.*`.
- [ ] Data quality checks pass; no synthetic fallback in production.
- [ ] Citation match rate ≥ 85% (papers > 3 months old).

**Analysis**
- [ ] Search benchmark (100+ queries) meets targets.
- [ ] Growth shown only where `sufficient_history` is true and stable.
- [ ] Leads robustness ≥ 70% and human-reviewed.

**Engineering**
- [ ] CI green, coverage ≥ 80%, lint/type clean.
- [ ] Load test: 20 concurrent users, search p95 < 1 s.

**Operations**
- [ ] Automated weekly refresh running for ≥ 4 weeks.
- [ ] Monitoring, error tracking, uptime checks, rollback procedure tested.

**Product**
- [ ] Dataset panel shows span, size, last refresh, release version.
- [ ] Limitations and disclaimers visible.
- [ ] Licence attribution in place.

---

## 4. Configuration changes summary

Proposed additions to `configs/config.yaml`:

```yaml
data:
  source: kaggle               # backfill; incremental updates use oai_pmh
  profile: standard            # dev | standard | full
  date_start: "2021-01-01"
  date_end: auto               # today
  categories: all_cs           # or an explicit list
  allow_synthetic_fallback: false
  min_span_months: 24
  sampling: stratified         # by primary_category x year_month

storage:
  backend: duckdb
  path: data/scholargrid.duckdb

enrich:
  openalex: true
  mailto: "you@example.com"
  refresh_schedule: {lt_3m: 7d, lt_24m: 30d, older: 90d}
  semantic_scholar: false

embedding:
  model_name: "all-MiniLM-L6-v2"   # re-evaluate vs specter2 / bge-small
  model_revision: "<pinned commit>"
  dtype: float16

index:
  backend: faiss_hnsw
  m: 32
  ef_search: 128

growth:
  min_papers_per_period: 30
  bootstrap_samples: 500
  require_stable: true

validation:
  gates:
    search_ndcg_at_10: 0.75
    min_span_months: 24
    max_noise_fraction: 0.35
```

---

## 5. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle snapshot or OAI-PMH unavailable | Stale data | Cache raw Parquet; refresh job retries; app shows staleness banner |
| OpenAlex rate limits / outages | Missing citations | Polite pool, backoff, incremental refresh, keep last good values |
| Clusters reshuffle every refresh | Trends meaningless | Cluster ID matching, scheduled (not weekly) full re-clusters |
| Hosting RAM limits at large scale | App crashes | mmap, ANN index, on-demand metadata, upgrade hardware tier |
| Embedding model upgrade breaks search | Bad results | Pin model revision; regression benchmark; versioned bundles |
| Users read leads as "proven gaps" | Misuse | Wording, badges, evidence, disclaimers, reliability gates |
| Scope creep | Delay | Phases 1–4 first; everything else after a working `standard` release |

---

## 6. Suggested order of execution (first 4 weeks)

1. **Week 1:** Fix the Kaggle loader, add `allow_synthetic_fallback: false`,
   download the snapshot, build a 3-year `dev` (5k) corpus. Set up `pytest`,
   `ruff`, CI. Write tests for ingestion and growth.
2. **Week 2:** DuckDB/Parquet storage, stratified `standard` (50k) corpus,
   incremental embeddings, FAISS index. Data quality checks.
3. **Week 3:** OpenAlex enrichment at scale with citation history; expanded
   search benchmark (100+ queries) and hybrid BM25 + dense search.
4. **Week 4:** Growth gating, bootstrap intervals, backtest; cluster ID
   matching; app updates for reliability badges and dataset panel; first
   `standard` release behind a staging URL.

After that: Phases 5, 7, 8, 9 in parallel, then the launch checklist.
