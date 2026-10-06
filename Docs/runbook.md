# ScholarGrid operations runbook

How to build, publish, roll back and debug ScholarGrid in production. Keep this
short and current; anything here should be runnable as written.

## Moving parts

| Piece | Where | Notes |
|---|---|---|
| Pipeline | `python pipeline/run_pipeline.py` | Stages: ingest, enrich, embed, analyze, validate, publish |
| Paper store | `data/raw/papers/` (Parquet) and `data/scholargrid.duckdb` | Upserts only; never hand-edit |
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
```

## Rolling back

```bash
python -m scholargrid.release list
python -m scholargrid.release rollback 2026.10.06       # repoints CURRENT
```

The app picks up the new `CURRENT` on its next restart. To freeze a release
regardless of new publishes, set `release.pin: "<id>"` in the config. Old
releases beyond `release.keep_last` are pruned automatically, and the CURRENT
and pinned releases are never pruned.

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

## Updating relevance judgments

```bash
python -m scholargrid.judgments export --out reports/judgments_todo.csv
# grade each row 0/1/2 by hand
python -m scholargrid.judgments import reports/judgments_todo.csv
```

Hand-graded judgments in `configs/search_judgments.json` replace the heuristic
grades for those queries.

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
