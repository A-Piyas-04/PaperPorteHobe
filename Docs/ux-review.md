# UX and architecture review

Reviewed and implemented locally on **2026-10-10**.

## Product intent

ScholarGrid helps students and researchers move from a broad computer-science interest to an evidence-backed reading direction. Its useful outcome is an informed reading list and understanding of nearby topics, not an automatically declared research gap. The proposal and SRS agree on this intent.

## What the code actually does

`scholargrid/pipeline.py` orchestrates ingest, enrichment, embeddings, analysis, validation, and release publication. Storage, incremental ingestion, hybrid BM25/dense retrieval, ANN indexing, growth gates, and versioned bundles already exist. The Streamlit application loads precomputed artifacts through `app/ui.py`. Expensive analysis remains offline. A React component provides the interactive Workspace, which now offers both a connected-node **Map** (Sigma v3 + Graphology, WebGL) and a **List** view over the same bounded payload, plus the paper detail, related work and saving.

The older technical reference describes a single Plotly map and an adaptive side panel. Before this change the actual application already had six pages and a Sigma graph component. The production roadmap's baseline statements (including no tests, 1,275 papers, and one week of history) are historical observations, not current facts. Current corpus and validation status must come from the loaded release.

## Findings and changes

| Finding | Implemented response |
|---|---|
| Six equally prominent destinations offered no clear starting sequence | Find papers, Workspace, and Reading list are primary; analytical tools live under More |
| Landing page explained search but not the practical outcome | A direct search action and a three-step interest → context → reading-list journey |
| No introduction once a dataset existed | Four-step introduction with Next, Back, Skip, completion, and replay on the home page |
| Dense full-corpus network with hover-driven highlighting | Rebuilt as a level-of-detail connected-node map (areas → area papers → paper neighbourhood) with click/keyboard selection; hover only previews a tooltip and never changes selection, colours or camera |
| Narrow layouts hid the detail pane | Responsive single-column list/detail navigation with Back to papers |
| Zero-result searches silently displayed the corpus | Explicit empty results and filter recovery guidance |
| Full-corpus search matches outside the visual sample disappeared | Required result indices are included in the bounded workspace payload |
| Paper index zero could not be preselected; index/string types disagreed | Numeric selection contract handles zero correctly |
| Hybrid ranking values appeared as percentage matches | Rank labels in the workspace; shared cards use an explicit ranking score when requested |
| Discovery ended without an actionable output | Session reading list with removal and CSV/BibTeX export, available from primary navigation |
| Long paper titles were truncated and 60 rows appeared at once | Unclamped titles, more reading space, and 20-paper progressive loading |
| Search could remain indefinitely in a busy state | Timeout guidance and retry, plus descriptive searching text |
| An earlier redesign removed the connected map, misreading the intent (the map was to be improved, not deleted) | Reinstated Sigma v3 + Graphology to power the restored map as a first-class Map view beside the List view; `npm audit` reports zero vulnerabilities |
| Search and metadata errors could overwhelm users | Existing friendly errors and honest data/degraded-search notices retained |

## The restored connected-node map

A previous amendment replaced the map with research-area cards. That misread the
intent: the map is a core feature and was meant to be *improved*, not removed.
This revision restores a real connected-node visualisation while keeping the
usability work (clear landing page, simplified nav, skippable/replayable tour,
area browsing, stable paper details, search, reading lists, exports, responsive
layouts). The v1 SRS 2D-plot requirement is honoured again, now as an explicit,
level-of-detail graph rather than a single flat scatter.

### Where it lives
The **Workspace** page offers two clearly labelled, first-class views that share
one selection: **Map** (default) and **List**. The map gets the width; a
collapsible detail drawer holds the active paper so three narrow columns are
avoided.

### Progressive levels of detail
- **Overview** — one bubble per research area, positioned at the area's embedding
  centroid, sized by rendered papers, connected by aggregated inter-area links.
  The full corpus is never drawn at once.
- **Focused area** — click a bubble to reveal that area's papers (bounded by
  `app.max_area_papers`, most-cited first) and the connections among them.
- **Selected paper** — click a node to emphasise its neighbourhood and supporting
  edges. *Focus this paper's connections* switches to a neighbourhood view, and
  *+ Expand* pulls in the next ring of neighbours.
- **Search** — matches are framed on the map with emphasis; every match is also
  reachable in the List view, including matches outside the browsing sample.

### Interaction rules (stability)
- Hover shows only a lightweight tooltip and a pointer cursor. It never changes
  the selection or detail pane, never recolours or redraws the whole map, never
  moves the camera, and never triggers a layout simulation or label flicker.
- Selection changes only through a deliberate click, a related-paper choice, a
  list row, or keyboard action, and persists until the next explicit change.
- The camera moves only on explicit controls (zoom in/out, Fit, Overview) or a
  deliberate related-paper jump — never on cursor movement, and never on an
  unrelated rerun such as saving a paper (the Sigma renderer persists across
  reruns, so the viewport is preserved).
- A single renderer is created per bundle and `kill()`ed on unmount; scope
  changes repopulate the same graph, so repeated navigation accumulates no
  renderers or listeners.

### Controls and legend
Zoom in/out, Fit current view, Return to overview, Clear selection, Back, and
(in a neighbourhood) Expand. A breadcrumb shows the current area/paper. A concise
legend explains that colour = area, size = citations or connections, and a link =
text similarity (and possibly shared references) — never a citation direction or a
confidence percentage. Positions come from the embedding projection; the legend
states that screen distance is only a rough similarity guide, not a measure of
research value or a gap.

### Accessibility and smaller screens
The List view is the keyboard-accessible path for selecting papers; the map
canvas carries an accessible name pointing to it, and all map controls are real
labelled buttons. On narrow screens the map uses a dedicated view with a bottom
sheet for details (search, related papers and Save stay reachable), and the List
view keeps its single-column list/detail with *Back to papers*. Animations and
camera moves honour `prefers-reduced-motion`. Colour is never the only selection
cue — the selected node also gets a ring and its row/area chip is marked.

### Fallbacks
If the browser cannot start the WebGL renderer, the map shows a clear message and
the List view remains a complete fallback. Empty searches, areas with no rendered
papers, and papers with no stored connections each show an explicit notice rather
than silently falling back to unrelated papers.

## Preserved interaction requirements

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
- Browsing uses the configured sample cap (`app.max_graph_nodes`, most-cited per area), and the map focuses an area with at most `app.max_area_papers` papers. Full-corpus search matches are always added to both the map and the List, and a visible disclosure distinguishes corpus, rendered and search-result counts. Verify the loaded release for current numbers; the local release checked here holds ~1,275 papers across 26 areas, not the larger historical snapshots some older notes mention. This is not a claim of validated large-corpus rendering performance.
- Area names remain automatically generated from keywords. The interface does not invent scientifically authoritative labels.
- The backend abstracts in the workspace are excerpts (up to 900 characters); arXiv provides the full original.
- Growth reliability, sparse-lead gating, citations, and research-quality limits remain governed by existing analysis and release metadata.
- Multi-user load testing, real-user usability sessions, production data quality, refresh operations, and deployment readiness are separate work. No deployment was performed.

## Validation

Completed locally:

- Python: **75 tests passed**, including page rendering for every page, tour progression/skipping/replay, search submission, saving/removing papers, empty area filters, and two new map-payload tests (aggregated area bubbles with centroids and honest corpus/rendered counts; a heavily-sampled bundle still keeps a search-matched paper reachable).
- Built-component browser regressions (Playwright/Chromium): **7 tests passed** — the map is the default and self-documenting view; list keyboard selection with stable hover and related work; empty search falls back to unrelated papers on neither map nor list; paper index-zero preselection plus mobile details open and return to the map; saving preserves the selection; search results replace an area focus and keep ranked order; selection is shared across map and list. Added to frontend CI.
- Full-app browser check on the loaded local release (~1,275 papers, 26 areas) via a real Chromium session: the Sigma map actually renders (WebGL canvas present, no fallback), the overview shows labelled area bubbles with aggregated links, selecting a paper shows its neighbourhood and detail drawer (abstract, metadata, arXiv link, related papers, Save), and the mobile layout opens details with no horizontal overflow. Screenshots in `reports/ux-review/` (`01-overview.png`, `02-selected-neighbourhood.png`, `03-mobile-details.png`).
- TypeScript/Vite production build succeeds and the committed `frontend/dist` is rebuilt so the normal Streamlit launch uses the restored map without a dev server. Ruff and mypy pass. `npm audit` reports **zero vulnerabilities**.

Use the commands in [Local development](next-instructions.md). These checks establish the tested local behaviour; they are not a claim of multi-user load capacity, large-corpus rendering performance, or usability outcomes for real researchers. Scalability beyond this release is bounded by design (level-of-detail + per-area caps), not demonstrated by visual inspection.
