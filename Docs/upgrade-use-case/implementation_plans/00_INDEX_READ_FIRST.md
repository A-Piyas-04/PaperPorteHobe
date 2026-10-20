# ScholarGrid → Research-Starter: Implementation Plan Series (READ FIRST)

> **How to use this series with Cursor**
> 1. Give Cursor **this file first** with the prompt in §1, and let it read the repository.
> 2. Then give it **one plan file at a time, in numeric order** (`01_…`, `02_…`, …).
> 3. Do not start file N+1 until every acceptance check in file N passes (`pytest`, `ruff check .`, `mypy`, and the manual checks listed).
> 4. Commit after each file, e.g. `git commit -m "plan-03: paper extraction stage"`.
>
> Every plan file is self-contained. Each one says what must already exist, what to build, the exact code, the config keys, the tests, and how to verify.

---

## 0. Why this series exists (product context, short)

The `refactor` branch of ScholarGrid is a well-engineered **semantic search + topic-cluster viewer** for arXiv CS papers. A product review (`ScholarGrid_Product_Review.md`, in the parent folder) found that it does **not** help early researchers with what they actually need:

1. understand a topic's sub-problems,
2. know **what to read first**,
3. **find real, evidence-backed research gaps** (the most important need),
4. check the gaps are not already solved,
5. turn a chosen gap into a **starter research plan**,
6. keep their work between sessions.

The current "Investigation leads" (`scholargrid/sparse.py`) find **outlier papers** in embedding space, not gaps. The pipeline only reads **abstracts**, and authors state open problems in **Limitations / Future Work** sections. Areas are global and coarse (77 for all of CS), and reading lists are session-only.

This series turns ScholarGrid into a **research-starter tool** while **keeping** its strong parts: the offline/online split, hybrid search, OpenAlex enrichment, the versioned releases and the statistical honesty.

---

## 1. Prompt to paste into Cursor together with this file

```text
You are implementing a multi-step upgrade of the ScholarGrid repository (Python 3.12,
Streamlit app + offline pipeline). Read 00_INDEX_READ_FIRST.md fully. Then read these
repository files so you know the existing contracts before writing anything:

  scholargrid/pipeline.py, scholargrid/config.py, scholargrid/schema.py,
  configs/config.yaml, configs/production.yaml, scholargrid/artifacts.py,
  scholargrid/enrich.py, scholargrid/sources/__init__.py, scholargrid/sources/openalex.py,
  scholargrid/live_search.py (class LiveSearcher), scholargrid/search.py (search()),
  scholargrid/embeddings.py (class Embedder), scholargrid/labeling.py,
  scholargrid/utils.py, app/streamlit_app.py, app/ui.py, app/views/home.py,
  app/views/library.py, tests/conftest.py, tests/test_app.py, Dockerfile,
  .github/workflows/ci.yml, pyproject.toml, requirements.txt.

Do NOT implement anything yet. Reply with:
 (a) a 10-line summary of the existing architecture in your own words,
 (b) any place where the conventions in 00_INDEX_READ_FIRST.md §4 disagree with the
     code you read (the code wins unless the plan says otherwise),
 (c) confirmation that tests currently pass (run: pip install -e ".[app,dev]" and pytest; do NOT
     install the heavy "ml" extra locally, see §4.7).
I will then give you plan files 01..11 one at a time.
```

---

## 2. The sequence

| # | File | What it delivers | Depends on | Effort (solo dev) |
|---|------|------------------|-----------|-------------------|
| 00 | `00_INDEX_READ_FIRST.md` | Conventions, contracts, glossary | none | n/a |
| 01 | `01_foundations_config_llm_layer.md` | New dependencies, config schema sections, the **free-tier LLM layer** (Gemini + OpenAI-compatible free cloud fallbacks: Groq / OpenRouter `:free`, structured output, quota manager with daily resume, request packing, disk cache, fake/null clients), domain-profile config | 00 | 2–3 days |
| 02 | `02_fulltext_acquisition.md` | New pipeline stage **`fulltext`**: fetch arXiv HTML → ar5iv → PDF, parse into sections, classify sections (limitations, future work, …), store and cache, plus a synthetic full-text generator for CI | 01 | 3–4 days |
| 03 | `03_paper_extraction.md` | New pipeline stage **`extract`**: per-paper structured facets (tasks, methods, datasets, metrics, domains, eval profile) and **stated open problems with verbatim quotes**; LLM batch + heuristic fallback; quote verification; facet canonicalization | 01, 02 | 4–5 days |
| 04 | `04_foundations_reading_path.md` | New pipeline stage **`foundations`**: citation backbone (co-cited seminal works incl. pre-2023 papers), survey detection, per-area reading paths, online reading-path builder | 01, 03 | 3 days |
| 05 | `05_topic_workspace.md` | Online **query-scoped Topic Workspace**: retrieval + live sources, fine-grained sub-topics (2 levels), c-TF-IDF + LLM labels and plain-language summaries, disk cache. New journey navigation shell and the **Topic** and **Read first** pages | 01–04 | 4–5 days |
| 06 | `06_gap_engine.md` | The **Gap Engine**: 6 detectors (stated problems, combinatorial, evaluation, unsettled results, citation bridges, long-standing), evidence verification, **live novelty check**, scoring, LLM card synthesis, offline stage **`gaps`** + online workspace mode, and the **Open problems** page | 01–05 | 7–9 days |
| 07 | `07_research_plan_generator.md` | Gap → **starter research plan** (question, hypotheses, baselines, datasets, metrics, 8–12-week milestones, risks, venues, people), grounded and exported (Markdown/JSON/BibTeX). **Plan** page | 06 | 3 days |
| 08 | `08_persistence_projects_auth.md` | **Durable projects**: SQLAlchemy (SQLite dev / Postgres prod), Alembic, optional Streamlit OIDC login, capability-link projects, notes/tags/status, import/export, **My project** page; migrate the session reading list | 01, 05–07 | 4 days |
| 09 | `09_evaluation_quality_telemetry.md` | **Evaluation**: rater UI (search, extraction, gap cards, plans), metrics (gap precision, novelty false-positive rate, kappa), free-model quality comparison harness, privacy-respecting telemetry and funnel, study protocol | 03, 06–08 | 3–4 days |
| 10 | `10_ux_integration_beginner_guidance.md` | Final UX: journey home page, trends → **opportunity quadrant**, retire old Leads, glossary/tooltips, empty states, accessibility, About/methodology rewrite, demo mode for new features | 05–09 | 3 days |
| 11 | `11_deployment_ops_cost_safety.md` | Deploy & operate: secrets, Dockerfile (built by Hugging Face, not locally), HF Spaces + free external Postgres (Neon), **online free-quota guard** (requests/day), BYOK (visitors' own free Gemini key), rate limiting, daily incremental refresh for new stages, licensing/attribution, runbook, launch checklist | all | 2–3 days |

**Do 01 → 11 in order.** Several files add code that later files import.

**Deferred items:** `LATER_NOT_NOW.md` lists what is deliberately left out of this series: local AI models
(Ollama), paid providers (xAI Grok, Gemini Pro / paid tier), paid hosting (own server + domain), and heavy local
tooling (Docker Desktop, local Postgres, Playwright, the `ml` extra, big local pipeline runs, `pymupdf`). Do not
implement any of it while working through 01–11.

**Parallel non-code work (do it alongside 01–03):** interview 8–10 early researchers and hand-make "Gap Cards" for 3 topics (see file 09 §8, "Discovery study protocol"). If the hand-made cards don't excite real students and supervisors, stop and rethink before building file 06.

---

## 3. Target architecture after all files

```
OFFLINE PIPELINE (python -m scholargrid.pipeline)
 ingest → enrich → fulltext → extract → embed → analyze → foundations → gaps → validate → publish
   │        │         │          │         │        │           │          │
   │        │         │          │         │        │           │          └─ gaps.json (area-level Gap Cards)
   │        │         │          │         │        │           └─ foundations.parquet, reading_paths.json
   │        │         │          │         │        └─ (existing) clusters, growth, graph, BM25 …
   │        │         │          │         └─ (existing) embeddings
   │        │         │          └─ extractions.parquet, paper_facets.parquet, problems.parquet,
   │        │         │             findings.parquet, facet_vocab.json
   │        │         └─ data/fulltext/sections.parquet (+ status)
   │        └─ (existing) OpenAlex citations/references
   └─ (existing) arXiv metadata

ONLINE APP (streamlit run app/streamlit_app.py)
 Start ─► Topic workspace ─► Read first ─► Open problems ─► Plan ─► My project
   │          │                  │              │              │         │
   │          │                  │              │              │         └─ SQL DB (projects, notes, gaps, plans)
   │          │                  │              │              └─ LLM (plan task), grounded in evidence
   │          │                  │              └─ detectors on workspace + offline cards + live novelty check
   │          │                  └─ co-citation backbone + surveys + OpenAlex lookups
   │          └─ LiveSearcher (collection + arXiv/OpenAlex/S2) → sub-topics → LLM labels
   └─ existing Search / Map / Areas / Trends / About stay under "More"
```

---

## 4. Global conventions (every plan file assumes these)

### 4.1 Code style (match the existing repo)
- Python **3.12** (`pyproject` says `>=3.11`; CI uses 3.12). Start every module with `from __future__ import annotations`.
- `ruff` line length **110**. The repo keeps `typing.List/Dict/Optional`, so follow the existing file's style.
- Logging: `from .utils import get_logger` → `log = get_logger("<stage>")`. Never `print` in library code (CLI `main()` functions may print).
- Config access: `cfg["section"]["key"]` (a `Config` wraps the validated dict). **Every new key must be added to `scholargrid/schema.py`**, because unknown keys are rejected (`extra="forbid"`).
- JSON IO: `scholargrid.utils.save_json / load_json`. Directory creation: `ensure_dir`.
- Timing: wrap stage work in `with stage_timer("<name>", ctx.timings) as info:` and put counts into `info`.
- Docstrings: one short module docstring that explains the *why*, like the existing modules.
- No new global singletons except module-level rate limiters, which the repo already uses (`RateLimiter` in `scholargrid/sources/__init__.py`).

### 4.2 Pipeline stage contract
- A stage is a function `stage_<name>(ctx: Context) -> None` in `scholargrid/pipeline.py`, registered in `STAGES`, `_FUNCS` and `_PROGRESS`.
- Stage inputs come from `ctx.frame(key)` / `ctx.json(key)` / `ctx.data[...]`, with on-disk fallbacks under `cfg.stages_dir`.
- Stages that **add files to the bundle** write into `cfg.processed_dir` (the staging bundle) after `analyze`. `publish` copies the whole directory, so new files ship automatically. Check this in `scholargrid/release.py::publish`: it must copy every file in `processed_dir`, not a fixed list. If it uses a fixed list, extend it.
- Expensive stages are **incremental and resumable**: they keep a per-paper cache keyed by a content hash, so re-running only processes new or changed papers.
- Every new stage has an `enabled` flag in config, and when disabled it must still write **empty but valid** outputs, so later stages and the app work.

### 4.3 Bundle contract
- New bundle files are **optional** for loading. `artifacts.load_bundle` must still load old releases that lack them (use `_load_optional` and empty DataFrames).
- New fields on `Bundle` (dataclass) get defaults (`field(default_factory=...)`).

### 4.4 LLM contract (built in file 01; used everywhere)
- **One gateway**: `from scholargrid.llm import get_llm` → `llm = get_llm(cfg)`. Never import a provider SDK (`google.genai`, `openai`) outside `scholargrid/llm/`.
- **In the app, always use `ui.get_llm()`** and pass the client into library functions (`build_workspace(..., llm=)`, `gaps_for_workspace(..., llm)`, `generate_plan(..., llm)`). That is how the online budget guard and bring-your-own-key (plan 11) take effect.
- **$0 API budget.** Every task runs on **free tiers**: Google Gemini (`gemini-3.5-flash-lite` for bulk tasks, `gemini-3.8-flash` for judgment-heavy tasks), with optional free cloud fallbacks through one OpenAI-compatible adapter (Groq free tier, OpenRouter `:free` models). Each task has an ordered `route` of `provider:model` entries in config. Paid providers and local models are out of scope (`LATER_NOT_NOW.md`). Never enable billing on the Google Cloud project behind the AI Studio key: without billing, going over quota only returns HTTP 429.
- The scarce resource is **requests per day**. The quota manager enforces your real limits (copied from the AI Studio rate-limit page), many small items are **packed** into one request (`structured_many`), and offline stages **stop cleanly when the quota is used up and resume on the next run**. There is no batch API on the free tier.
- Structured output only: every call returns a **pydantic model** validated from JSON-constrained output, with one automatic repair retry.
- Every call goes through the **disk cache**, the **quota manager** and the **usage ledger**. Tests use `FakeLLM` (`llm.provider: fake`). `provider: none` makes every LLM feature degrade to its heuristic path; `provider: live` calls the configured free providers.
- Free-tier content may be used by the provider to improve its products. Send only public paper text and anonymous queries; warn users before sending their own text (plan 07 background field).
- Paper text is **untrusted data**. Always wrap it in tags and tell the model to treat it as data, never instructions.

### 4.5 Testing contract
- Tests are offline: `tests/conftest.py::make_config(tmp_path)` already forces synthetic data, tfidf/pca/kmeans backends and no live search. New plans extend `make_config` to set `llm.provider: fake` and `fulltext.source: synthetic`.
- Every new module gets a `tests/test_<module>.py`. Every new page gets added to `PAGES` in `tests/test_app.py`.
- No test may hit the network. Use `monkeypatch` for HTTP and `FakeLLM` for LLM.
- Commands that must pass after every plan file:
  ```bash
  ruff check .
  mypy
  pytest
  python -m scholargrid.pipeline --config configs/ci.yaml --no-publish   # end-to-end offline
  streamlit run app/streamlit_app.py   # manual smoke test of the pages named in that file
  ```
- All of these are light and safe to run on a laptop. Anything heavier follows §4.7.

### 4.6 Data honesty contract (keep the team's best habit)
- Every displayed claim is traceable to papers (`arxiv_id` / DOI link) and, for gaps, to a **verbatim quote** that was verified against the source text.
- Every Gap Card shows **"Why this might not be a gap"** and its **novelty-check** result and date.
- Counts are always shown next to scores. Unknown is shown as unknown, never as zero.

### 4.7 Laptop-light rule (where things run)
The product is a website used in the browser. The developer's laptop writes code and runs **light** checks only;
heavy work runs on free cloud machines.

| Runs on the laptop | Runs in GitHub Actions (free) | Runs on Hugging Face Spaces (free CPU) |
|---|---|---|
| `ruff`, `mypy`, `pytest` (offline, FakeLLM, synthetic data) | the `dev-build` workflow (plan 01 §8.2) for real-data checks | the Streamlit app for visitors |
| `python -m scholargrid.pipeline --config configs/ci.yaml --no-publish` | the weekly refresh and daily extraction (plan 11 §9) | the Docker image build (from the `Dockerfile`) |
| tiny real checks with `configs/domain_nlp_tiny.yaml` (≈ 20 papers, a few LLM requests) | Postgres tests (CI service container) | |
| `streamlit run app/streamlit_app.py` on the CI/tiny bundle or a release downloaded from the `dev-build` artifact | anything with thousands of papers, MiniLM embeddings, or hours of full-text fetching | |

- Local install: `pip install -e ".[app,dev,llm,fulltext,db]"`. **Do not install the `ml` extra locally**
  (sentence-transformers → PyTorch, several GB). `embedding.backend: auto` falls back to TF-IDF without it.
- No Docker Desktop, no local Postgres, no local AI models, no browser-automation installs.
- If a plan file's acceptance check needs real data at scale, run it through the `dev-build` workflow and
  download the artifact; don't run it locally.

---

## 5. Shared names (contracts across files, don't rename)

| Name | Defined in | Used by |
|------|-----------|---------|
| `scholargrid/llm/` package: `get_llm`, `build_router`, `LLMRouter` (`structured`, `structured_many`, `map_structured`), `LLMResult`, `LLMError`, `LLMRefusal`, `LLMUnavailable`, `QuotaExhausted`, `BudgetExceeded`, `FakeLLM`, `strict_json_schema` | 01 | 03–07, 09, 11 |
| `scholargrid/llm/prompts.py`: `Prompt`, `PROMPTS`, `render(name, **vars)` | 01 (registry), each feature adds its prompts | 03–07 |
| `scholargrid/fulltext/`: `FulltextStore`, `run_fulltext`, `SectionKind` | 02 | 03, 06 |
| `scholargrid/extract/`: `PaperExtraction`, `StatedProblem`, `ExtractionStore`, `run_extract`, `canonicalize_facets` | 03 | 04–07, 09 |
| `scholargrid/foundations.py`: `WorkLookup`, `build_foundations`, `reading_path` | 04 | 05, 07 |
| `scholargrid/workspace/`: `TopicWorkspace`, `build_workspace`, `Subtopic` | 05 | 06, 07, 08, 10 |
| `scholargrid/gaps/`: `GapCard`, `Evidence`, `NoveltyCheck`, `run_offline_gaps`, `gaps_for_workspace` | 06 | 07, 08, 09, 10 |
| `scholargrid/plan/`: `ResearchPlan`, `generate_plan`, `plan_to_markdown` | 07 | 08, 09 |
| `scholargrid/persistence/`: `get_repo`, `ProjectRepo`, ORM models | 08 | 09, 10, 11 |
| `scholargrid/evaluation/`: sampling, `metrics`, compare harness; `app/telemetry.py`: `track(event, **props)`, `session_id()` | 09 | 10, 11 |
| Bundle files: `extractions.parquet`, `paper_facets.parquet`, `problems.parquet`, `findings.parquet`, `facet_vocab.json`, `foundations.parquet`, `reading_paths.json`, `gaps.json` | 03, 04, 06 | app |
| Pages (`app/views/`): `start`, `topic`, `reading`, `gaps`, `plan`, `project`, `evaluate` | 05–09 | nav |

---

## 6. New dependencies (installed in file 01; listed here for overview)

| Package | Why | Where | License |
|---|---|---|---|
| `google-genai` (1.x) | Gemini API, free tier (structured JSON output) | pipeline + app | Apache-2.0 |
| `openai` (client library only) | OpenAI-compatible endpoints: Groq, OpenRouter (no OpenAI account needed) | pipeline + app | Apache-2.0 |
| `beautifulsoup4` | Parse arXiv / ar5iv HTML | pipeline | MIT |
| `lxml` | Fast HTML parser backend for bs4 | pipeline | BSD |
| `pypdf` | PDF text fallback (permissive license) | pipeline | BSD-3 |
| `rapidfuzz` | Verify quotes are verbatim (fuzzy substring) and match facet aliases | pipeline + app | MIT |
| `sqlalchemy` (2.x) | Durable projects DB | app | MIT |
| `alembic` | DB migrations | app/ops | MIT |
| `psycopg[binary]` *(optional extra)* | Postgres driver for hosted DB | app | LGPL-3.0 (dynamic linking OK) |
| `Authlib` *(optional extra)* | Required by Streamlit `st.login()` OIDC | app | BSD-3 |

Already present and reused: `numpy`, `pandas`, `scikit-learn`, `scipy`, `pydantic`, `requests`, `pyarrow`, `streamlit`, `plotly`, `sentence-transformers`, `umap-learn`, `hdbscan`, `duckdb`.

**Version pinning rule:** before adding a pin, run `pip index versions <package>`, choose the newest release that is **at least two weeks old**, pin it exactly in `requirements.txt`, and put a compatible range in `pyproject.toml`. Then run `pip-audit -r requirements.txt`.

---

## 7. Environment variables / secrets (full list after all files)

| Variable | Used by | Required? |
|---|---|---|
| `GEMINI_API_KEY` | LLM layer (pipeline + app), free key from Google AI Studio | Required when `llm.provider: live` |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY` | Optional free fallback providers | Optional |
| `OPENALEX_MAILTO` | Existing enrichment + new foundations lookups (polite pool) | Strongly recommended |
| `S2_API_KEY` | Existing Semantic Scholar live search / OA PDF lookup | Optional |
| `SCHOLARGRID_DB_URL` | Persistence (file 08). Default `sqlite:///data/app.db` | Required in production (Postgres) |
| `SCHOLARGRID_ADMIN_EMAILS` | Admin/eval pages (files 09, 11) | Optional |
| `SCHOLARGRID_EVAL_MODE` | Shows the Evaluate page (file 09) | Optional |
| `SENTRY_DSN` | Existing error tracking | Optional |
| `SCHOLARGRID_CONFIG`, `SCHOLARGRID_ENV`, `SCHOLARGRID_RELEASE` | Existing | Existing |
| `.streamlit/secrets.toml` `[auth]` block | Streamlit OIDC login (file 08) | Optional |

Never commit secrets. `.env` is already gitignored. File 01 adds `.env.example`.

---

## 8. Glossary (used throughout)

- **Area**: existing global cluster from `analyze` (77 in the current release).
- **Topic workspace**: the set of papers retrieved for one user query, re-clustered into **sub-topics** (file 05).
- **Facet**: a structured attribute extracted from a paper: task, method, dataset, metric, domain, language, modality (file 03).
- **Stated open problem**: a limitation or future-work statement **written by the paper's authors**, stored with a verbatim quote (file 03).
- **Gap candidate**: a detector output before verification (file 06).
- **Gap Card**: a verified, novelty-checked, scored candidate with a plain-language statement, evidence, "why it might not be a gap", feasibility and first steps (file 06).
- **Novelty check**: a live search across arXiv/OpenAlex/Semantic Scholar plus an LLM judgment of whether recent papers already address the gap (file 06).
- **Reading path**: an ordered list of surveys → foundations → key recent papers → frontier, with a one-line reason each (file 04).

---

## 9. Definition of done for the whole series

- A beginner types a topic and, within ~1 minute, sees: sub-topics with plain summaries, an ordered reading path including foundational pre-2023 papers, and at least 5 Gap Cards with verified quotes, novelty results and feasibility. They can pick one, generate a plan, save it to a durable project, and export it.
- `pytest`, `ruff`, `mypy` and the CI pipeline run pass in CI without network or API keys.
- Nothing in the series costs money, and nothing heavy runs on the developer's laptop (§4.7).
- File 09's evaluation shows gap precision ≥ 60% and novelty false positives ≤ 15% on at least 3 topics rated by domain experts. If not, iterate on files 03 and 06 before launch.
