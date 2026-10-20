# ScholarGrid operations runbook

How to build, publish, roll back and debug ScholarGrid in production. Keep this
short and current; anything here should be runnable as written.

## Moving parts

| Piece | Where | Notes |
|---|---|---|
| Pipeline | `python pipeline/run_pipeline.py` | Stages: ingest, enrich, embed, analyze, validate, publish |
| Paper store | `data/raw/arxiv/` (Parquet, system of record) and `data/scholargrid.duckdb` (working copy) | Upserts only; never hand-edit. **Offline only**: the app never reads either, so neither goes into a deployment |
| Stage cache | `data/stages/` | Safe to delete; the next run rebuilds it |
| Releases | `data/releases/<YYYY.MM.DD[.n]>/` plus `CURRENT` | What the app serves |
| App | Docker image (`Dockerfile`) or `streamlit run app/streamlit_app.py` | Reads `CURRENT` (or `release.pin`) |
| Weekly refresh | `.github/workflows/weekly-refresh.yml` | Opens a GitHub issue if it fails |

## Configuration and secrets

- Production uses `configs/production.yaml` (Kaggle snapshot) for the initial
  build and `configs/refresh.yaml` (OAI-PMH) for weekly updates. Select one with
  `--config` or `SCHOLARGRID_CONFIG`.
- Secrets come from environment variables only. Never put them in YAML or Git:
  - `OPENALEX_MAILTO`: contact email for the OpenAlex polite pool (strongly recommended;
    without it large enrichment runs get rate-limited).
  - `SENTRY_DSN`: enables error tracking for the pipeline and app (optional).
  - `S2_API_KEY`: Semantic Scholar cross-check (optional).
- `SCHOLARGRID_ENV=production` turns on the production checks: no synthetic
  fallback, quality failures stop the build, and failing required validation
  gates block publishing.

## First production build

1. Download `arxiv-metadata-oai-snapshot.json` from Kaggle into `data/raw/`.
2. Run `python pipeline/run_pipeline.py --config configs/production.yaml`.
3. Read `reports/validation_report.md`. Every required gate must pass, otherwise
   nothing is published.
4. Check `data/releases/CURRENT` and the release's `manifest.json`, then start
   the app.

## Rebuilding one stage

The stages cache their outputs, so you can rerun from the point that changed:

```bash
python pipeline/run_pipeline.py --stage validate          # re-score only
python pipeline/run_pipeline.py --from-stage analyze      # re-cluster + everything after
python pipeline/run_pipeline.py --no-publish              # build and validate, don't publish
python pipeline/run_pipeline.py --stage publish --force-publish   # publish despite failed gates
```

`--force-publish` is for deliberate exceptions (for example, a search gate
measured only against heuristic grades). The release manifest records
`validation.gates_overridden: true` so the override is visible later.

## Rolling back

```bash
python -m scholargrid.release list
python -m scholargrid.release rollback 2026.10.06       # repoints CURRENT
```

The app picks up the new `CURRENT` on its next restart. To freeze a release
regardless of new publishes, set `release.pin: "<id>"` in the config. Old
releases beyond `release.keep_last` are pruned automatically, and the CURRENT
and pinned releases are never pruned.

## Disk housekeeping (offline machine)

What the app needs is only `data/releases/<id>/` plus `data/releases/CURRENT`
(and `data/live/` for papers found by live search). Everything else under
`data/` is offline working state:

| Path | Size (typ.) | Safe to delete? |
|---|---|---|
| `data/scholargrid.duckdb` | hundreds of MB | Yes, with `python -m scholargrid.storage prune`. The Parquet raw layer keeps every paper and the DuckDB copy is rebuilt from it on the next harvest |
| `data/stages/` | tens of MB + embedding cache | Yes; the next run rebuilds it. Keep `data/stages/embedding_cache/` if you can, otherwise every paper is re-embedded |
| `data/raw/arxiv/` | hundreds of MB | No, unless you plan to re-harvest from scratch |
| `data/raw/openalex_cache.parquet` | tens of MB | Yes, but the next enrich re-queries OpenAlex for every paper |

Never copy `data/scholargrid.duckdb` (or anything outside `data/releases/`) into
an app image or onto the server.

## When the weekly refresh fails

The previous release stays live, so users are not affected right away. The app
shows a stale-data banner once the data is older than `monitoring.stale_after_days`.

1. Open the failed run from the `refresh-failure` issue and download the
   `refresh-report` artifact.
2. Find the cause by looking at the JSON log lines (`"level": "ERROR"`) and the stage timings:
   - **arXiv OAI-PMH outage / 503s:** the harvester retries and saves its
     progress in `data/raw/oai_state.json`. Rerun the workflow later and it
     resumes where it stopped.
   - **OpenAlex errors / rate limits:** check that `OPENALEX_MAILTO` is set. Papers
     that weren't enriched keep their cached values and are retried next run.
   - **Data quality gate:** see `checks` in the report (span, per-month counts,
     category share, row-count change). A large row-count drop usually means a
     partial harvest, so don't relax thresholds to force it through.
   - **Validation gate:** compare with the previous run in
     `reports/validation_history.jsonl`. Fix the cause, or publish the previous
     good config.
3. Close the issue with a note about what happened.

## Free LLM keys

All LLM features run on free tiers ($0 API spend). Put keys in `.env` (see `.env.example`)
or in GitHub Actions secrets; never in YAML.

| Provider | Where | Env var | Notes |
|---|---|---|---|
| **Gemini** (required) | Google AI Studio → "Get API key" | `GEMINI_API_KEY` | Free tier. Check your limits at AI Studio → rate limits. One project = one quota; extra keys in the same project don't add quota. **Do not enable billing** on that project. |
| Groq (optional) | console.groq.com → API keys | `GROQ_API_KEY` | Free tier, per-organization limits shown in the console. Don't add a payment method. |
| OpenRouter (optional) | openrouter.ai → keys | `OPENROUTER_API_KEY` | Only models ending in `:free` cost nothing. Don't buy credits. |

`llm.provider` picks the mode: `live` calls the providers, `fake` is for tests, `none` makes
every LLM feature use its heuristic fallback. `python -m scholargrid.llm check` is the first
diagnostic when something LLM-related misbehaves.

## Running on free quota

1. Copy your real limits from AI Studio into `llm.limits` (the defaults are guesses).
2. Run `python -m scholargrid.llm estimate --task extract --n <scope size>` (light; runs anywhere).
3. Let GitHub Actions run the pipeline daily (the `dev-build` workflow, started once a day, until the
   daily workflow exists). Each run extracts until the quota is exhausted, then **publishes anyway**,
   with heuristic extraction for the rest. The next run continues where it stopped. Don't run large
   pipelines on the laptop.
4. To go faster: add Groq or OpenRouter `:free` models to the `extract` route, raise `pack_size`
   (4 → 6; check that truncations stay rare in `python -m scholargrid.llm usage`), or narrow `scope`.

```bash
python -m scholargrid.llm quota      # used today / rpd per route, exhausted-until, next reset
python -m scholargrid.llm usage      # requests and tokens by task and route this month
```

## Playbooks (free-tier operation)

| Situation | Symptom | Action |
|---|---|---|
| Daily free quota used up | "AI allowance used up" notes; `llm quota` shows exhausted routes | Expected on busy days. Point users to their own key (More → Your AI key). Lower `gaps.online_max_new_cards` / `gaps.novelty_top_judge`; check the LLM cache persists (a cache-miss storm wastes quota) |
| Pipeline extraction progressing slowly | `extract stats` shows many papers waiting for an LLM pass | Normal on free tiers. Add Groq or OpenRouter `:free` to the `extract` route, raise `pack_size` if truncation is rare, or narrow `scope`. Don't move the pipeline to the laptop |
| GitHub Actions minutes running out | Scheduled jobs stop or are queued | Make the repository public (unlimited minutes), or lower `fulltext.max_papers_per_run` and run the daily job less often |
| Space asleep | The first visitor waits about a minute | Expected on the free CPU tier (sleeps after 48 h without visitors). No action |
| Provider limits changed | Unexpected 429s at low counts | Re-read the limits in AI Studio / the provider console; update `llm.limits` |
| A model was shut down or renamed | `BadRequest` / 404 in the logs for one route | Update `llm.tasks.*.route` to a current free model. Bump the prompt versions if output quality changes |
| Gemini outage | `LLMError` spikes; template results | Nothing urgent (the app degrades). If prolonged, set `SCHOLARGRID_LLM_PROVIDER=none` to avoid timeouts |
| Postgres down | Saving fails | The app shows "Couldn't save"; reading still works. Check the provider status and restore from backup |
| arXiv 429/403 in full text | `fulltext stats` errors climb | Stop; never lower the interval; resume later |
| Bad gap quality after a release | Evaluation precision drops / "Not useful" spikes | Roll back (`python -m scholargrid.release rollback <id>`), compare the evaluation round, fix thresholds/prompts, bump prompt versions |
| Leaked project link | A user reports it | My project → Settings → Create a new link |
| Leaked API key | A key appears in logs, a repo or a screenshot | Delete the key in AI Studio / the console immediately, create a new one, update the secrets |

Logs and Sentry events are scrubbed of API keys, project tokens (`?k=`) and database passwords
(`scholargrid/redact.py`), but treat any leak as real.

## Recorded measurements

Fill these in as they are measured (plans 02, 03, 05, 06, 09):

- Full-text coverage on real cs.CL (`fulltext stats`, dev-build): _not yet measured_
- Extraction manual QA (10 papers: real problems %, exact quotes %, missed problems): _not yet measured_
- Topic workspace timings for 3 real topics (`ws.stats["timings"]`): _not yet measured_
- Gap-card spot check (3 topics × 10 cards: real & open / solved / nonsense): _not yet measured_
- Extraction route decision (`compare-extract`): _not yet decided_
- Latest gap round and extraction round headlines: _not yet run_

## Updating relevance judgments

```bash
python -m scholargrid.judgments export --sample 15 --out reports/judgments_todo.csv
# fill the empty `grade` column by hand: 0 = not relevant, 1 = partly, 2 = relevant
python -m scholargrid.judgments import reports/judgments_todo.csv
python pipeline/run_pipeline.py --stage validate
```

Hand-graded judgments in `configs/search_judgments.json` replace the heuristic
grades for those queries. Until some exist, every search number in
`validation_report.md` is marked "heuristic grades": it measures whether results
fall in the expected arXiv categories and contain expected keywords, not whether
a person would find them relevant. Once 10 or more queries are judged, the report
shows a separate "hand-judged queries only" line; record that as the baseline
and set `validation.gates.search_ndcg_at_10` to a realistic value from it.

## Security notes

- **Only load bundles you built yourself.** Bundles contain joblib files (and older
  ones contain pickle files), and loading a joblib or pickle file can run
  arbitrary code.
- The container runs as a non-root user and serves only Streamlit on port 8501.
  The health check is `/_stcore/health`.
- Users never see raw tracebacks (`showErrorDetails = "none"`). Errors are logged,
  and sent to Sentry when `SENTRY_DSN` is set.
- Run `pip-audit -r requirements.txt` and review Dependabot PRs every week.

## Data licensing

arXiv metadata is CC0, and abstracts are shown with a link back to arXiv. OpenAlex
data is CC0. The app footer credits both. Don't redistribute full texts.
