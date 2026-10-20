# Plan 11: Deployment, operations, free-quota control and safety

> **Prerequisites:** plans 01–10 are merged and the evaluation bars from plan 09 are met (or you consciously launch as a "beta").
> **Produces:**
> - a production container with all new dependencies, migrations and secrets handling,
> - **external Postgres** for projects,
> - an **online free-quota guard** (requests per day, globally and per session) with graceful degradation,
> - **bring-your-own-key** (each visitor can paste their *own* free Gemini key, the main way to scale a free app),
> - per-session rate limits and an LLM kill switch,
> - secret redaction in error reports,
> - a **daily** incremental pipeline run that continues extraction where yesterday's quota stopped,
> - licensing and attribution, runbook playbooks and a launch checklist.
>
> **Why:** the whole system runs on **$0 API spend**. A public app on one shared free key would run out of requests within minutes on a busy day. Free hosting also has ephemeral disks, so projects need an external database.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 11_deployment_ops_cost_safety.md. Order: §3 kill switch + env overrides → §4 online
quota guard → §5 bring-your-own-key → §6 rate limits + concurrency → §7 secret redaction → §8 container
+ start script → §9 scheduled workflows → §10 CI additions → §11 licensing/terms → §12 docs (deployment,
runbook, launch checklist) → §13 tests. Never log API keys or project tokens. Finish with §14.
```

---

## 2. Production topology

```
                ┌──────────── GitHub Actions (free; NOT the laptop) ────────────────────────────┐
                │ weekly:  ingest → enrich → fulltext → extract → embed → analyze → foundations → │
                │          gaps → validate → publish                                              │
                │ daily:   --from-stage fulltext (continue full text + extraction on fresh quota) │
                └─────────────────────────────────────┬───────────────────────────────────────────┘
                                                      │ upload release bundle to the Space (§9.3, HF_TOKEN)
                                                      ▼
 users' browsers ──HTTPS──► Hugging Face Space (CPU basic, free; HF builds the Docker image)
                    streamlit app ── Gemini free tier (server key: small shared daily allowance)
                         │      ├── visitor's own free Gemini key (BYOK, own quota)
                         │      └── arXiv / OpenAlex / Semantic Scholar (live search, novelty check)
                         └──── Postgres (Neon free tier)
                               projects, ratings, events, daily LLM request counts
```

**Hosting is Hugging Face Spaces only.** `Docs/deployment.md` Option A (a rented server, a domain and Caddy) is
paid and deferred to `LATER_NOT_NOW.md`. The developer's laptop only writes code and runs light checks
(00 §4.7): no Docker Desktop, no local Postgres, no big pipeline runs.

**What runs without any LLM at all** (offline results in the bundle, keyword labels, template cards and plans): everything. Free LLM quota adds plain-language summaries, better extraction, synthesized Gap Cards and personalised plans.

Secrets:
- **Pipeline (GitHub repository secrets):** `GEMINI_API_KEY` (optional `GROQ_API_KEY`, `OPENROUTER_API_KEY`), `OPENALEX_MAILTO`, `S2_API_KEY`, `HF_TOKEN` (a Hugging Face write token for §9.3).
- **App:** `GEMINI_API_KEY` (server allowance), `SCHOLARGRID_DB_URL`, `OPENALEX_MAILTO`. Optional: `SENTRY_DSN`, `[auth]` secrets, `SCHOLARGRID_ADMIN_EMAILS`.

**Quota sharing:** limits are per Google Cloud project, and the pipeline and the app share one project if they use the same key. Schedule the pipeline to start **right after the daily reset** (midnight Pacific), cap it with `llm.budget.offline_max_requests`, and keep `llm.budget.online_daily_requests` for the app. Together they must stay below your project's real RPD. Do not create extra accounts or projects to multiply free quota; that can break the provider's terms.

---

## 3. Kill switch and env overrides (`scholargrid/config.py`)

```python
    "SCHOLARGRID_LLM_PROVIDER": ("llm", "provider"),                       # live | fake | none
    "SCHOLARGRID_DAILY_LLM_REQUESTS": ("llm", "budget", "online_daily_requests"),
    "SCHOLARGRID_OFFLINE_MAX_REQUESTS": ("llm", "budget", "offline_max_requests"),
    "SCHOLARGRID_PERSISTENCE_AUTH": ("persistence", "auth"),
```

`_apply_env` writes strings, and pydantic's lax mode coerces numeric strings for `int` fields. Add a test.

`SCHOLARGRID_LLM_PROVIDER=none` (restart only, no redeploy): labels become keywords, new cards and plans use templates, and novelty uses heuristics. Offline cards in the bundle still show. A banner says "AI features are temporarily off; results use simpler methods."

---

## 4. Online quota guard (server key)

The plan 01 `QuotaManager` already enforces the provider's per-minute and per-day limits on one machine. The online guard adds **product-level** limits that hold across processes and instances: the app may use at most `online_daily_requests` per day (leaving the rest of the project's quota to the pipeline) and `online_session_requests` per browser session (one visitor can't use up everyone's allowance).

### 4.1 Model + migration (persistence)

```python
class LlmUsageDaily(Base):
    __tablename__ = "llm_usage_daily"
    __table_args__ = (UniqueConstraint("day", "scope"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10), index=True)        # YYYY-MM-DD in America/Los_Angeles (Gemini reset)
    scope: Mapped[str] = mapped_column(String(80), index=True)      # "global" or "session:<sid>"
    requests: Mapped[int] = mapped_column(Integer, default=0)
```

Repo methods:
- `requests_today(scope) -> int`
- `add_request(scopes: list[str])`: an atomic upsert. On Postgres use `INSERT … ON CONFLICT (day, scope) DO UPDATE SET requests = llm_usage_daily.requests + 1`; on SQLite, use `INSERT OR IGNORE` + `UPDATE` in one transaction.

### 4.2 Session context

`scholargrid/llm/context.py`:

```python
import contextvars

current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("sg_session", default=None)
```

The app sets `current_session.set(telemetry.session_id())` at the top of every script run (`streamlit_app.main`). The plan 01 router already copies the context into worker threads in `map_structured`; `structured` and `structured_many` run in the caller's thread.

### 4.3 Guard

```python
class OnlineGuard:
    """Product limits for the SERVER key: raise BudgetExceeded before a request over the daily or session
    allowance; count after. Not installed on bring-your-own-key routers."""

    def __init__(self, cfg, repo):
        self.daily = int(cfg["llm"]["budget"]["online_daily_requests"])
        self.session = int(cfg["llm"]["budget"]["online_session_requests"])
        self.repo = repo

    def check(self, task: str) -> None:
        sid = current_session.get()
        if self.repo.requests_today("global") >= self.daily:
            raise BudgetExceeded("daily shared AI allowance used up")
        if sid and self.repo.requests_today(f"session:{sid}") >= self.session:
            raise BudgetExceeded("session AI allowance used up")

    def record(self, task: str, route: str) -> None:
        sid = current_session.get()
        self.repo.add_request(["global"] + ([f"session:{sid}"] if sid else []))
```

Wire it in `app/ui.py`:

```python
@st.cache_resource
def _shared_llm():
    cfg = get_config()
    llm = scholargrid.llm.get_llm(cfg)
    repo = get_repo()
    if repo is not None and llm.provider == "live":
        guard = OnlineGuard(cfg, repo)
        llm.ledger.max_requests = None              # online: no per-process run cap; the guard applies
        llm.ledger.extra_guard = guard.check
        llm.ledger.on_record = guard.record
    return llm


def get_llm():
    """BYOK router for this session if the visitor added a key (§5), else the shared, guarded one."""
    return st.session_state.get("_byok_llm") or _shared_llm()
```

**UI behaviour** on `BudgetExceeded` / `QuotaExhausted`: every caller already falls back (keyword labels, template cards and plans). Show one friendly note per session:

> "Today's free AI allowance is used up, so you're seeing simpler results. To continue with full features, add your own free Gemini API key (takes about a minute; see Settings), or come back after the daily reset (around {local time})."

Compute the reset time from `QuotaManager.next_reset(route)` and show it in the visitor's timezone via `st.context.timezone` if available, else UTC.

---

## 5. Bring your own key (BYOK): the main way to scale

- A Settings expander (in My project → Settings, and in the More menu) with a short how-to:
  1. Open Google AI Studio and sign in.
  2. Click "Get API key" and create a key.
  3. Paste it here.

  Then the field `st.text_input("Your Gemini API key", type="password")`, with the help text: "Used only for your requests in this browser session. Never stored or logged. Requests count against **your** free quota. Free-tier content may be used by Google to improve its products."
- Validate with the plan 01 `check` prompt (1 request) before accepting.
- Store only in session state. Build a per-session router:
  ```python
  udir = os.path.join(cfg.abspath(cfg["llm"]["usage_dir"]), "byok", hashlib.sha256(key.encode()).hexdigest()[:12])
  st.session_state["_byok_llm"] = scholargrid.llm.build_router(cfg, api_keys={"gemini": key}, usage_dir=udir)
  ```
  The hashed folder gives the visitor's key its own quota counters (it is a hash, not the key). No online guard is installed: it is their quota.
- The answer cache is shared (same `cache_dir`). That is fine and saves everyone quota, since cached answers come from public paper text. Topic text and plan backgrounds are part of cache keys, not readable values.
- "Remove key" deletes it from the session.
- Disabled when `llm.allow_byok` is false.
- Optional: also accept Groq / OpenRouter keys the same way (`api_keys={"groq": ...}`), which enables those routes for that session.

---

## 6. Rate limits and concurrency

`app/limits.py`:

```python
LIMITS = {"workspace_build": (10, 3600), "gaps_generate": (4, 3600), "plan_generate": (4, 3600),
          "novelty_check": (15, 3600), "project_create": (10, 3600)}
_SEMAPHORE = threading.BoundedSemaphore(2)       # at most 2 expensive builds per process at once


def allow(action: str) -> bool:
    """Sliding-window limit per browser session (timestamps in st.session_state)."""


@contextmanager
def heavy_slot(timeout: float = 30.0):
    """Waits for a free slot; on timeout raises Busy so the page can say 'Many users right now, try again shortly'."""
```

Apply them before every expensive action and around `build_workspace`, `gaps_for_workspace` and `generate_plan`. Cached results bypass both. Make the limits configurable under `app.limits`. BYOK sessions get 3× the limits; they spend their own quota.

---

## 7. Error reporting without secrets (`scholargrid/monitoring.py`)

```python
_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                 # Google API keys (Gemini)
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),                    # Groq
    re.compile(r"sk-or-[0-9A-Za-z\-]{20,}"),                # OpenRouter
    re.compile(r"hf_[0-9A-Za-z]{20,}"),                     # Hugging Face tokens
    re.compile(r"([?&](?:k|key)=)[A-Za-z0-9_\-]+"),         # project tokens, ?key= query params
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:]+:)[^@]+@"),
]


def _scrub(text: str) -> str:
    for rx in _SECRET_PATTERNS:
        text = rx.sub(lambda m: (m.group(1) if m.groups() else "") + "[redacted]", text)
    return text


def _before_send(event, hint):
    import json

    return json.loads(_scrub(json.dumps(event, default=str)))
```

Pass `before_send=_before_send` to `sentry_sdk.init`, and add a `logging.Filter` on the `scholargrid` root logger that applies `_scrub` to messages and args.

---

## 8. Container and start script

### 8.1 `Dockerfile`
- The builder installs from `requirements.txt` (now including google-genai, openai, rapidfuzz, sqlalchemy, alembic, psycopg[binary], Authlib, bs4, lxml, pypdf) plus `sentry-sdk`.
- `COPY --chown=scholargrid:scholargrid alembic.ini ./` and `COPY --chown=scholargrid:scholargrid scripts/start.sh ./scripts/start.sh`.
- `CMD ["bash", "scripts/start.sh"]`.

### 8.2 `scripts/start.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${STREAMLIT_SECRETS_TOML_B64:-}" ]]; then
  mkdir -p .streamlit
  echo "$STREAMLIT_SECRETS_TOML_B64" | base64 -d > .streamlit/secrets.toml
  chmod 600 .streamlit/secrets.toml
fi
python -m scholargrid.persistence upgrade
exec streamlit run app/streamlit_app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
```

### 8.3 Hosted database (free tier)
**Neon** Postgres free tier (recommended; Supabase's free tier pauses inactive projects, which would break saving): create a database and put its URL into the secret `SCHOLARGRID_DB_URL` (`postgresql://user:pass@host/db?sslmode=require`). Don't add a payment method. Use the provider's backups. Add a weekly `pg_dump` in GitHub Actions to an artifact retained for 30 days, and document the restore steps. Keep data small (telemetry `retention_days`, the weekly `purge-events`) to stay inside the free storage limit.

### 8.4 Hugging Face Spaces (free CPU), the only deployment target
Hugging Face builds the image from the `Dockerfile` **on its own servers** whenever the Space repo changes. There
is no local `docker build` in this series (Docker Desktop is deferred, `LATER_NOT_NOW.md`). Secrets: `GEMINI_API_KEY`, `SCHOLARGRID_DB_URL`, `OPENALEX_MAILTO`, `S2_API_KEY`, `SENTRY_DSN` (optional), `SCHOLARGRID_ADMIN_EMAILS`, `STREAMLIT_SECRETS_TOML_B64` (optional login). The ephemeral disk is fine: projects are in Postgres, `data/workspaces/` and `data/llm_cache/` are caches, and the release bundle ships with the upload (existing deployment docs). Restarts reset the local quota counters, but the online guard counts in Postgres, and the provider's own limit is the hard backstop.

---

## 9. Scheduled workflows

### 9.1 Weekly full refresh (`.github/workflows/weekly-refresh.yml`)
1. Add secrets to `env`: `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}` (and optional `GROQ_API_KEY`, `OPENROUTER_API_KEY`), plus `HF_TOKEN` / `HF_SPACE` for §9.3.
2. Extend the `actions/cache` paths:
   ```yaml
          path: |
            data/raw
            data/stages
            data/releases
            data/fulltext
            data/extract
            data/llm_cache
            data/llm_usage
   ```
3. Schedule it shortly **after the Gemini reset**: midnight Pacific is 07:00 UTC in summer and 08:00 UTC in winter, so use `cron: "30 8 * * 1"`.

### 9.2 Daily continuation (`.github/workflows/daily-extract.yml`, new)

```yaml
name: Daily extraction (free quota)
on:
  schedule:
    - cron: "45 8 * * 2-7"     # every day except the weekly-refresh day, after the Pacific-midnight reset
  workflow_dispatch:
jobs:
  continue:
    runs-on: ubuntu-latest
    timeout-minutes: 180
    env:
      SCHOLARGRID_CONFIG: configs/refresh.yaml
      SCHOLARGRID_ENV: production
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      OPENALEX_MAILTO: ${{ secrets.OPENALEX_MAILTO }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.txt
      - uses: actions/cache@v4
        with:
          path: |
            data/raw
            data/stages
            data/releases
            data/fulltext
            data/extract
            data/llm_cache
            data/llm_usage
          key: scholargrid-data-${{ github.run_id }}
          restore-keys: scholargrid-data-
      - run: python pipeline/run_pipeline.py --from-stage fulltext
      - run: python -m scholargrid.extract stats && python -m scholargrid.llm quota
```

`configs/refresh.yaml` additions:
- `fulltext.max_papers_per_run: 400` (≈ 20 min at the polite 3-second rate).
- `llm.budget.offline_max_requests`: your project's RPD minus `online_daily_requests`, minus a margin.

**Actions minutes (free):** unlimited on a **public** repository. A private repository on a free account gets
2,000 minutes/month; the weekly refresh plus a daily ~40-minute run can exceed that. GitHub does not charge
without a payment method (jobs just stop), but to keep everything running either make the repository public
(secrets stay hidden) or lower `fulltext.max_papers_per_run` to ~150 and run the daily job every other day.

The run publishes a new release each day with more LLM extractions and cards. Previous days' work is cached, so nothing is paid or requested twice. Keep the existing "open an issue on failure" step in both workflows.

### 9.3 Publish the release to the Space from Actions (no laptop upload)
Add to the job-level `env` of **both** workflows (a step's own `env` can't be used in its `if:`):

```yaml
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      HF_SPACE: ${{ vars.HF_SPACE }}              # repository variable, e.g. "<hf-user>/scholargrid"
```

Then append this step after the pipeline step. It uploads only the current release and `CURRENT` into the
Space's `data/releases/` and deletes older releases there in the same commit, so the Space repo stays small. The
Space rebuilds automatically. Without the secret the step is skipped.

```yaml
      - name: Publish release to the Hugging Face Space
        if: success() && env.HF_TOKEN != '' && env.HF_SPACE != ''
        run: |
          pip install -U huggingface_hub
          rel="$(cat data/releases/CURRENT)"
          mkdir -p _space_releases
          cp -r "data/releases/$rel" _space_releases/
          cp data/releases/CURRENT _space_releases/
          hf upload "$HF_SPACE" _space_releases data/releases --repo-type=space \
            --delete "*" --commit-message "release $rel"
```

Code changes still reach the Space the documented way (`Docs/deployment.md` §B3–B5): that is a small upload of
source files from the laptop, not a build. If the installed `huggingface_hub` names the CLI or the `--delete`
flag differently, adapt the command and keep the behaviour.

---

## 10. CI additions (`.github/workflows/ci.yml`)

- `test` job: `pip install -e ".[app,dev,llm,fulltext,db]" duckdb faiss-cpu`. Tests stay offline (FakeLLM, stubbed provider clients, synthetic full text, SQLite).
- An optional `postgres` job with a `postgres:16` service container running `tests/test_persistence.py` with `SCHOLARGRID_DB_URL=postgresql://postgres:pw@localhost:5432/postgres`.
- A guard test that only `scholargrid/llm/` imports provider SDKs:
  ```python
  def test_only_llm_package_imports_provider_sdks():
      import pathlib, re
      rx = re.compile(r"^\s*(import|from)\s+(google\.genai|google\s+import\s+genai|openai)\b", re.M)
      for p in pathlib.Path("scholargrid").rglob("*.py"):
          if "scholargrid/llm" in p.as_posix():
              continue
          assert not rx.search(p.read_text()), p
  ```
- `pip-audit` and `npm audit` unchanged.

---

## 11. Licensing, terms and attribution

- **Gemini API:** read and follow the Gemini API Additional Terms of Service and Google's usage policies. Check the conditions for the **free tier** (e.g. age and regional requirements, and that free-tier content may be used to improve Google's products), and whether serving a public app from the free tier is allowed in your situation. Disclose on the About page: "Summaries, Gap Card wording and plans are generated with Google Gemini (free tier) and grounded in the cited papers. Free-tier content may be used by Google to improve its products; don't enter personal information."
- **Groq / OpenRouter models:** follow each provider's terms and each open model's licence when you enable them.
- **arXiv:** keep the existing attribution line. Respect automated-access rate limits (plan 02 enforces 3 s). Show only short quotes (≤ 300 chars) with links; full text stays in the unpublished `data/fulltext/` store.
- **OpenAlex:** CC0 data. **Semantic Scholar:** follow the API licence (attribution, no bulk redistribution).
- **Generated-content disclaimer** (About + plan export footer): "Generated text can be wrong. Verify every claim against the cited papers."
- Privacy notice (plans 09/10) and a contact address for corrections and takedowns.
- PDF parsing uses `pypdf` (BSD) only; AGPL-licensed `pymupdf` is deferred (`LATER_NOT_NOW.md`).

---

## 12. Documentation updates

### 12.1 `Docs/deployment.md`: new section "9. Research-starter features (free tier)"
- The secrets table (00 §7), how to get each free key (plan 01 §3.4), and copying real limits from AI Studio into `llm.limits`.
- Postgres setup (Neon step by step).
- OIDC login setup (optional).
- The quota-sharing rule (§2), the daily and weekly workflows, publishing to the Space from Actions (§9.3, the `HF_TOKEN` secret and `HF_SPACE` variable), the BYOK explanation, and the kill switch.
- At the top of the existing "Choose a hosting option" table, add: "This project uses **Option B** only. Option A is paid and deferred; see `Docs/upgrade-use-case/implementation_plans/LATER_NOT_NOW.md`." Remove "(recommended)" from Option A.

### 12.2 `Docs/runbook.md`: playbooks

| Situation | Symptom | Action |
|---|---|---|
| Daily free quota used up | "AI allowance used up" notes; `llm quota` shows exhausted routes | Expected on busy days. Point users to BYOK. Lower `online_max_new_cards` / `novelty_top_judge`; check the cache is persisting (a cache-miss storm wastes quota) |
| Pipeline extraction progressing slowly | `extract stats` shows many `pending_llm` | Normal on free tiers. Add Groq or OpenRouter `:free` to the `extract` route, raise `pack_size` if truncation is rare, or narrow `scope`. Don't move the pipeline to the laptop |
| GitHub Actions minutes running out | Scheduled jobs stop or are queued; billing page shows minutes used | Make the repository public (unlimited minutes), or lower `fulltext.max_papers_per_run` and run the daily job less often |
| Space asleep | The first visitor waits about a minute | Expected on the free CPU tier (sleeps after 48 h without visitors). No action |
| Provider limits changed | Unexpected 429s at low counts | Re-read the limits in AI Studio / the provider console; update `llm.limits` |
| A model was shut down or renamed | `BadRequest` / 404 in the logs for one route | Update `llm.tasks.*.route` to a current free model (check the Gemini models page). Bump the prompt versions if output quality changes |
| Gemini outage | `LLMError` spikes; template results | Nothing urgent (the app degrades). If it is prolonged, set `SCHOLARGRID_LLM_PROVIDER=none` to avoid timeouts |
| Postgres down | Saving fails | The app shows "Saving is temporarily unavailable"; reading still works. Check the provider status and restore from backup |
| arXiv 429/403 in full text | `fulltext stats` errors climb | Stop; never lower the interval; resume later |
| Bad gap quality after a release | Evaluation precision drops / "Not useful" spikes | Roll back the release (`python -m scholargrid.release rollback <id>`), compare the evaluation round, fix the thresholds/prompts, bump the prompt versions |
| Leaked project link | A user reports it | Settings → Rotate link |
| Leaked API key | A key appears in logs, a repo or a screenshot | Delete the key in AI Studio / the console immediately, create a new one, and update the secrets |

### 12.3 Launch checklist (append to the existing one)

- [ ] Plan 09 gap round: precision ≥ 60%, novelty false positives ≤ 15%, recorded
- [ ] Extraction round: problem precision ≥ 80%, using the route chosen in plan 09 §8
- [ ] 5 unaided user walkthroughs done; top issues fixed
- [ ] Real limits copied into `llm.limits`; the server key works on the Space (AI features on, no `LLMUnavailable` in the Space logs)
- [ ] Nothing paid: no billing on the Google Cloud project, no payment method on Groq / OpenRouter / Neon / Hugging Face / GitHub
- [ ] Publishing from Actions to the Space (§9.3) worked once
- [ ] The pipeline cap + online allowance stay below the project's RPD; the daily workflow ran once successfully
- [ ] Postgres configured, migrations applied, a backup restore tested
- [ ] Kill switch tested (`SCHOLARGRID_LLM_PROVIDER=none` → app works); "allowance used up" path tested
- [ ] BYOK tested end to end with a fresh free key; the key never appears in logs or Sentry
- [ ] Rate limits tested
- [ ] About page: methodology, limitations, privacy (incl. free-tier data use), contact, licences, AI disclosure
- [ ] Mobile check at 375 px; keyboard-only walkthrough
- [ ] Demo mode works on a fresh clone without any key

---

## 13. Tests

1. `tests/test_online_guard.py` (SQLite):
   - the global allowance reached → `BudgetExceeded`,
   - session allowances are independent per `current_session`,
   - `add_request` from 20 threads counts 20,
   - the guard is **not** installed on a BYOK router (`build_router(..., api_keys=...)`),
   - `map_structured` propagates `current_session` into worker threads.
2. `tests/test_byok.py`: the BYOK router uses its own `usage_dir` (quota counters don't touch the shared ones); the key never appears in the usage ledger or captured logs; removing it restores the shared router.
3. `tests/test_limits.py`: the sliding window allows N then blocks; `heavy_slot` raises `Busy` on timeout.
4. `tests/test_monitoring.py`: `_scrub` redacts `AIza…`, `gsk_…`, `sk-or-…`, `hf_…`, `?key=…`, `?k=…` and Postgres passwords; the log filter scrubs messages.
5. `tests/test_config_env.py`: `SCHOLARGRID_LLM_PROVIDER=none` → `cfg["llm"]["provider"] == "none"`; `SCHOLARGRID_DAILY_LLM_REQUESTS=50` → int 50.
6. The provider-SDK import guard from §10.

---

## 14. Acceptance checks

On the laptop (light):

```bash
ruff check . && mypy && pytest -q
streamlit run app/streamlit_app.py      # SQLite, CI/demo bundle: the "allowance used up" note and BYOK settings render
```

On the free cloud (no Docker or Postgres on the laptop):

1. Push. CI runs the tests, including the `postgres` service-container job.
2. Create a second, **private** free Space as staging (`Docs/deployment.md` §B1–B5) with the secrets from §8.4,
   pointing `SCHOLARGRID_DB_URL` at the Neon database, plus the variable `SCHOLARGRID_DAILY_LLM_REQUESTS=20`.
   Hugging Face builds the image from the `Dockerfile`; watch the build in the Space's **Logs** tab.
3. In the browser: the full journey until the 20-request allowance runs out → friendly note → add your own key
   in Settings → full features again.
4. Kill switch: set the Space variable `SCHOLARGRID_LLM_PROVIDER=none`, restart the Space → the app still works
   with simpler results.
5. Run the daily workflow once by hand (`workflow_dispatch`) and confirm §9.3 updated the Space's release.

Then deploy to the public Space (`Docs/deployment.md` Option B), run the launch checklist, and announce as a **beta** with a feedback link.
