# UX and architecture review

Reviewed and implemented locally on **2026-10-10**.

## Product intent

ScholarGrid helps students and researchers move from a broad computer-science interest to an evidence-backed reading direction. Its useful outcome is an informed reading list and understanding of nearby topics, not an automatically declared research gap. The proposal and SRS agree on this intent.

## What the code actually does

`scholargrid/pipeline.py` orchestrates ingest, enrichment, embeddings, analysis, validation, and release publication. Storage, incremental ingestion, hybrid BM25/dense retrieval, ANN indexing, growth gates, and versioned bundles already exist. The Streamlit application loads precomputed artifacts through `app/ui.py`. Expensive analysis remains offline. A React component provides the interactive reading workspace.

The older technical reference describes a single Plotly map and an adaptive side panel. Before this change the actual application already had six pages and a Sigma graph component. The production roadmap's baseline statements (including no tests, 1,275 papers, and one week of history) are historical observations, not current facts. Current corpus and validation status must come from the loaded release.

## Findings and changes

| Finding | Implemented response |
|---|---|
| Six equally prominent destinations offered no clear starting sequence | Find papers, Workspace, and Reading list are primary; analytical tools live under More |
| Landing page explained search but not the practical outcome | A direct search action and a three-step interest → context → reading-list journey |
| No introduction once a dataset existed | Four-step introduction with Next, Back, Skip, completion, and replay on the home page |
| Dense full-corpus network with hover-driven highlighting | Replaced by searchable area cards and a paper list plus stable detail view; no network or cursor-driven selection |
| Narrow layouts hid the detail pane | Responsive single-column list/detail navigation with Back to papers |
| Zero-result searches silently displayed the corpus | Explicit empty results and filter recovery guidance |
| Full-corpus search matches outside the visual sample disappeared | Required result indices are included in the bounded workspace payload |
| Paper index zero could not be preselected; index/string types disagreed | Numeric selection contract handles zero correctly |
| Hybrid ranking values appeared as percentage matches | Rank labels in the workspace; shared cards use an explicit ranking score when requested |
| Discovery ended without an actionable output | Session reading list with removal and CSV/BibTeX export, available from primary navigation |
| Long paper titles were truncated and 60 rows appeared at once | Unclamped titles, more reading space, and 20-paper progressive loading |
| Search could remain indefinitely in a busy state | Timeout guidance and retry, plus descriptive searching text |
| Graph renderer and dependencies became unused after the redesign | Removed Sigma/Graphology; updated Vite to 6.4.4 after checking the dependency audit |
| Search and metadata errors could overwhelm users | Existing friendly errors and honest data/degraded-search notices retained |

## Updated interaction requirements

This revision supersedes the v1 SRS requirement that the primary landscape UI be a 2D plot. Embedding coordinates and analytical graph artifacts remain available to the offline analysis. The user-facing landscape is now a directory of research areas and explicit paper-to-paper connections.

1. The first page states the user's benefit and offers an obvious search action.
2. A newcomer can skip the tour without losing access to the app, and replay it later.
3. A paper's detail remains stable when the cursor moves.
4. Related work is revealed for the selected paper instead of drawing all links simultaneously.
5. Search with no matches must not show unrelated papers as results.
6. Details and original-paper links remain accessible at phone widths and with a keyboard.
7. Evidence, methodology, dataset coverage, and uncertainty remain discoverable.
8. Users can preserve useful results with an export.

## Boundaries and follow-up

- Reading lists and tour completion are **session-scoped**, with clear wording. Accounts and durable cross-device storage are not implemented.
- Browsing uses the configured sample cap, with full-corpus search matches added. The sample disclosure remains visible. The app still loads the underlying bundle; this is not a claim of validated 500k-paper performance.
- Area names remain automatically generated from keywords. The interface does not invent scientifically authoritative labels.
- The backend abstracts in the workspace are excerpts (up to 900 characters); arXiv provides the full original.
- Growth reliability, sparse-lead gating, citations, and research-quality limits remain governed by existing analysis and release metadata.
- Multi-user load testing, real-user usability sessions, production data quality, refresh operations, and deployment readiness are separate work. No deployment was performed.

## Validation

Completed locally:

- Python: **73 tests passed**, including page rendering, tour progression/skipping/replay, search submission, saving/removing papers, and empty area filters.
- Built-component browser regressions: **5 tests passed** for click/keyboard selection, hover stability, empty results, index-zero preselection, mobile navigation, saving, and search/filter reset. Added to frontend CI.
- Full-app browser check on the local 50,000-paper release: search, saving from both interfaces, reading-list navigation, BibTeX download, area browsing, related-paper reading, mobile details/back navigation, component search, and no horizontal overflow or browser errors.
- TypeScript/Vite production build, Ruff, and mypy passed. `npm audit` reported **zero vulnerabilities** after the toolchain update.

Use the commands in [Local development](next-instructions.md). Local review screenshots are under `reports/ux-review/` when generated; those artifacts are not required to run the app. These checks establish the tested local behavior, not multi-user load capacity or usability outcomes for real researchers.
