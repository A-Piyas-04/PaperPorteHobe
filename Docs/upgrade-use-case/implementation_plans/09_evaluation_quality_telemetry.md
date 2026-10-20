# Plan 09: Evaluation, quality measurement and telemetry

> **Prerequisites:** plans 03 and 06–08 are merged (extractions, Gap Cards, plans, database).
> **Produces:**
> 1. a **rater UI** (`Evaluate` page, gated) for search results, extractions, Gap Cards, reading paths and plans, with blind, stratified, overlapping samples,
> 2. `scholargrid/evaluation/metrics.py`: gap precision, novelty false-positive rate, confidence calibration, per-type quality, inter-rater agreement (Cohen's κ), extraction precision, plan scores, plus a Markdown/JSON report,
> 3. a **free-model comparison harness** for extraction (quality vs requests per paper per route), so choosing the bulk-extraction route is a measured decision,
> 4. privacy-respecting **telemetry** with a funnel dashboard,
> 5. the **study protocol** for discovery interviews and the benchmark against "Google Scholar + a general LLM".
>
> **Why:** the product promise ("real, open, feasible gaps") is an empirical claim. Without measurement, the team can't tell whether changes help, and can't honestly launch.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 09_evaluation_quality_telemetry.md. Ratings and events go to the plan-08 database
via new ORM models + an Alembic revision. The Evaluate page is hidden unless SCHOLARGRID_EVAL_MODE=1
or the signed-in user is in SCHOLARGRID_ADMIN_EMAILS. Raters must NOT see scores/confidence/novelty
verdicts while rating (blind). Telemetry never stores raw query text unless telemetry.store_queries
is true. Order: §3 config → §4 models + migration → §5 sampling → §6 Evaluate page → §7 metrics + report
→ §8 model comparison → §9 telemetry (app/telemetry.py) → §10 dashboard → §11 docs → §12 tests. Finish with §13.
```

---

## 2. What is measured, and the launch bar

| Object | Question | Metric | Launch bar |
|---|---|---|---|
| Gap Card | Is it a genuine research problem *and* still open? | **gap precision** = share rated real=yes and open≠no | **≥ 60%** |
| Gap Card (status `novel`) | Was it actually already solved? | **novelty false-positive rate** = share open=no | **≤ 15%** |
| Gap Card | Do confidence labels mean something? | precision by confidence (high > medium > low) | monotone |
| Stated problem (extraction) | Is it really stated by the authors, and faithfully paraphrased? | extraction precision | ≥ 80% |
| Facets | Are tasks/datasets correct? | precision; recall on a 20-paper gold set | ≥ 85% precision |
| Search | Relevance of top 10 | nDCG@10 with **human** grades (finishes REF-04) | baseline recorded |
| Plan | Realistic, grounded, useful | mean 1–5 scores | ≥ 3.5 each |
| Product | Do users get value? | funnel + 7-day return | tracked |

---

## 3. Config

```python
class Evaluation(_Section):
    enabled: bool = True
    eval_mode_env: str = "SCHOLARGRID_EVAL_MODE"
    admin_emails_env: str = "SCHOLARGRID_ADMIN_EMAILS"
    overlap_share: float = Field(default=0.2, ge=0, le=1)     # items rated by >= 2 raters
    sample_seed: int = 7
    report_dir: str = "reports/eval"


class Telemetry(_Section):
    enabled: bool = True
    store_queries: bool = False          # raw query text only with this explicitly on
    retention_days: int = Field(default=180, ge=7)
    sample_rate: float = Field(default=1.0, ge=0, le=1)
```

Add `evaluation` and `telemetry` sections. `make_config` in tests: `telemetry.enabled = False` by default. Tests that check telemetry enable it explicitly.

---

## 4. Models + migration (`scholargrid/persistence/models.py`)

```python
class Rating(Base):
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rater: Mapped[str] = mapped_column(String(80), index=True)            # rater code or user id
    item_type: Mapped[str] = mapped_column(String(30), index=True)        # gap_card|problem|facets|search|plan|reading
    item_id: Mapped[str] = mapped_column(String(80), index=True)
    batch: Mapped[str] = mapped_column(String(80), default="", index=True)  # evaluation round name
    context: Mapped[dict] = mapped_column(JsonType, default=dict)          # topic, scope, bundle_version, model
    criteria: Mapped[dict] = mapped_column(JsonType, default=dict)         # rubric answers
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    session: Mapped[str] = mapped_column(String(32), index=True)          # random per browser session
    user_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(50), index=True)
    props: Mapped[dict] = mapped_column(JsonType, default=dict)
```

Run `alembic revision --autogenerate -m "ratings and events"` and review the result.

---

## 5. Sampling: `scholargrid/evaluation/sampling.py`

Evaluation rounds are **frozen samples**. A round is created once, stored as JSON in `reports/eval/<batch>.json`, and raters work through it in a randomized order.

```python
def make_gap_round(name: str, cards: List[Dict], n: int, overlap: float, seed: int) -> Dict:
    """Stratify by (type, confidence) proportionally with a floor of 2 per stratum when available;
    pick `overlap` share of items to be shown to every rater; shuffle per rater at display time."""


def make_extraction_round(name: str, bundle, n_papers: int, seed: int) -> Dict:
    """n papers with verified problems; items = each problem + one 'facets' item per paper."""


def make_search_round(name: str, cfg, n_queries: int) -> Dict:
    """Reuse scholargrid.judgments.sample_queries + export logic; items = (query, result paper)."""
```

CLI:

```text
python -m scholargrid.evaluation round gaps --name gaps-2026-11 --topics "rag hallucination; code llms; low-resource mt" --n 60
python -m scholargrid.evaluation round extraction --name ext-2026-11 --n 25
python -m scholargrid.evaluation round search --name search-2026-11 --queries 15
python -m scholargrid.evaluation report --batch gaps-2026-11
```

For `round gaps` with `--topics`: build each workspace (plan 05) and its cards (plan 06) via the library functions, not the UI, then sample from the union. Store the **full card JSON** in the round file, so later code changes don't change what was rated.

---

## 6. The Evaluate page: `app/views/evaluate.py`

Visibility: register the page only if `os.environ.get(cfg["evaluation"]["eval_mode_env"]) == "1"` or the signed-in email is in `SCHOLARGRID_ADMIN_EMAILS` (comma-separated).

Flow:
1. **Rater identity:** a rater code (`st.text_input`, e.g. "R1-mahin"), stored in the session. Signed-in users default to their email.
2. **Choose round:** select from `reports/eval/*.json`.
3. **Progress:** "17 of 60 rated", with "skip" and "back".
4. **Item view (blind):** show only what a user would see *minus* scores, confidence, novelty verdicts and detector statistics.

Gap Card item:
- title, statement, why it matters, evidence (quotes + links), first steps, why it might not be a gap,
- rubric (all required unless marked optional):
  - **Real research problem?** yes / partly / no
  - **Still open, to your knowledge?** yes / unsure / no. If no: "link to a paper that solves it" (optional text)
  - **Specific enough to start?** 1 vague · 2 workable · 3 concrete
  - **Feasible for a master's student in ~3 months?** yes / maybe / no
  - **Does the evidence support the claim?** yes / partly / no
  - **Would you suggest it to a student?** yes / no
  - comment (optional)

Extraction **problem** item: the paper title + link + section, the quote, the statement. Rubric: "Stated by the authors as open/limitation?" yes/no; "Statement faithful to the quote?" yes/partly/no; "Specificity" 1–3.

Extraction **facets** item: the paper title + abstract + extracted tasks/methods/datasets/metrics as checkboxes. The rater unticks wrong ones and types missing ones (comma-separated).

**Search** item: query + result title/abstract → grade 0/1/2 (the same definitions as `scholargrid/judgments.py`). Also write the grade to `configs/search_judgments.json` through `judgments import` logic, so validation uses human grades (REF-04).

**Plan** item: the full plan markdown → realistic scope 1–5, grounded in references 1–5, first steps useful 1–5, factual errors (text).

**Reading path** item: the ordered list → "Right papers?" 1–5, "Sensible order?" 1–5, "Anything essential missing?" (text).

5. Save each answer immediately (`Rating` row with `batch`, `context` = the topic/scope/bundle version/model from the item).

---

## 7. Metrics and report: `scholargrid/evaluation/metrics.py`

```python
def gap_metrics(ratings: pd.DataFrame, round_items: Dict[str, Dict]) -> Dict:
    """ratings: item_id, rater, criteria(dict). round_items: item_id -> card JSON (for type/confidence/novelty)."""
    df = _flatten(ratings)                         # columns real, open, specific, feasible, supported, suggest
    df["good"] = (df["real"] == "yes") & (df["open"] != "no")
    per_item = df.groupby("item_id")["good"].mean() >= 0.5          # majority across raters
    meta = pd.DataFrame.from_dict(round_items, orient="index")
    out = {"n_items": int(per_item.size), "gap_precision": round(float(per_item.mean()), 3)}
    novel = meta.index[meta["novelty"].map(lambda n: n.get("status")) == "novel"]
    open_no = df.groupby("item_id")["open"].agg(lambda s: (s == "no").mean() >= 0.5)
    out["novelty_false_positive_rate"] = round(float(open_no.reindex(novel).dropna().mean()), 3) if len(novel) else None
    for key in ("type", "confidence"):
        out[f"precision_by_{key}"] = {k: round(float(per_item.reindex(g.index).dropna().mean()), 3)
                                      for k, g in meta.groupby(key)}
    out["feasible_yes_rate"] = round(float((df["feasible"] == "yes").mean()), 3)
    out["kappa_real"] = pairwise_kappa(df, "real")
    out["kappa_open"] = pairwise_kappa(df, "open")
    return out


def pairwise_kappa(df: pd.DataFrame, field: str) -> Optional[float]:
    """Mean Cohen's kappa (sklearn.metrics.cohen_kappa_score) over rater pairs on overlapping items."""
```

Also write `extraction_metrics` (problem precision, faithful rate, facet precision/recall from the facets items), `search_metrics` (call `validation.search_metrics` with human grades) and `plan_metrics` (means and SDs).

`report(batch)` writes `reports/eval/<batch>.md` and `.json`: headline metrics against the launch bar (✅/❌), the per-type table, the calibration table, kappas, and the 10 worst-rated items with rater comments (the most useful part for fixing things). Append a line to `reports/eval/history.jsonl` for trends across rounds.

Add an **advisory** validation gate: if the latest gap round's precision is older than 60 days, the validation report says "Gap quality last measured N days ago".

---

## 8. Free-model comparison harness (extraction): `scholargrid/evaluation/compare.py`

Purpose: decide with data **which free route** to use for bulk extraction. For example: is `gemini-3.5-flash-lite` (more requests per day) good enough compared with `gemini-3.8-flash`, and is Groq's free `openai/gpt-oss-120b` a usable fallback? Only free cloud routes are compared; local models are deferred (`LATER_NOT_NOW.md`).

```text
python -m scholargrid.evaluation compare-extract \
    --routes gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite,groq:openai/gpt-oss-120b --n 20 \
    --config configs/domain_nlp_tiny.yaml
```

This is laptop-light: the work is API requests, and the tiny config already has full text for its 20 papers.
For a larger sample (`--n 30+` on `configs/domain_nlp_dev.yaml`), add the command as a step in the `dev-build`
workflow (plan 01 §8.2) and download `reports/eval/` from the artifact.

Steps:
1. Sample `n` in-scope papers with full text (fixed seed). Keep `n` small (≈ 30): with `pack_size` 4 that is ≈ 8 requests per route. Check `python -m scholargrid.llm quota` first.
2. For each route: copy the config, set `llm.tasks.extract.route = [route]`, and run the extraction on the sample. The cache key includes the route, so runs don't collide, and a rerun the next day continues where quota stopped.
3. Automatic metrics per route, with the strongest route (`gemini-3.8-flash`) as the reference:
   - verified-quote rate,
   - problems per paper,
   - Jaccard of canonical task/dataset sets vs the reference,
   - share of reference problems matched (statement cosine ≥ 0.8),
   - **requests per paper** (packing efficiency), truncation rate, and seconds per paper.
4. Create a **blind** rating round (`extraction` items) mixing all routes' outputs, with the route hidden in the UI but stored in `context`.
5. Report `reports/eval/compare-extract-<date>.md` with a table, plus the decision rule written in the report:

> Use the higher-quota route for bulk extraction only if, versus the reference, the human-rated problem precision is within **5 percentage points**, the verified-quote rate is ≥ **95%** of the reference's, and the task/dataset Jaccard is ≥ **0.7**. Otherwise keep the stronger model first in the route and accept slower progress (more days). Record the decision and the numbers in `Docs/runbook.md`.

## 9. Telemetry: `app/telemetry.py`

Telemetry needs Streamlit session state and the app's cached repository, so it lives in the **app**, not in the `scholargrid` library (the library must never import `ui`).

```python
"""Product analytics: anonymous, minimal, opt-out. Never blocks the UI."""
from __future__ import annotations

import hashlib
import random
import secrets
import threading

import streamlit as st

EVENTS = {"app_open", "topic_started", "workspace_built", "reading_viewed", "gaps_generated",
          "gap_evidence_opened", "gap_saved", "gap_not_useful", "novelty_checked", "plan_generated",
          "plan_exported", "project_created", "project_returned", "paper_saved", "export_downloaded"}


def session_id() -> str:
    if "_sid" not in st.session_state:
        st.session_state["_sid"] = secrets.token_hex(16)
    return st.session_state["_sid"]


def query_props(cfg, query: str) -> dict:
    props = {"q_hash": hashlib.sha256(query.lower().encode()).hexdigest()[:16],
             "q_words": len(query.split())}
    if cfg["telemetry"]["store_queries"]:
        props["q"] = query[:300]
    return props


def track(name: str, **props) -> None:
    import ui

    cfg = ui.get_config()
    t = cfg["telemetry"]
    if not t["enabled"] or st.session_state.get("telemetry_opt_out") or name not in EVENTS:
        return
    if random.random() > float(t["sample_rate"]):
        return
    repo = ui.get_repo()
    if repo is None:
        return
    row = {"session": session_id(), "user_id": st.session_state.get("user_id"),
           "project_id": st.session_state.get("proj_id"), "name": name, "props": props}
    threading.Thread(target=_safe_insert, args=(repo, row), daemon=True).start()


def _safe_insert(repo, row: dict) -> None:
    try:
        repo.add_event(**row)
    except Exception:
        pass          # analytics must never break the app
```

Add `repo.add_event(...)` and `repo.purge_events(older_than_days)` to `ProjectRepo`.

Instrumentation points:
- Start submit → `topic_started` with `query_props`.
- Workspace done → `workspace_built` (n_papers, k, seconds, cached).
- Reading page view → `reading_viewed`.
- Gaps generated → `gaps_generated` (n_cards, seconds, requests_used).
- Evidence expander → `gap_evidence_opened` (gap_id, type).
- Save gap → `gap_saved`.
- "Not useful" → `gap_not_useful`, with a reason from {already done, too vague, not feasible, wrong/irrelevant, other}.
- Plan → `plan_generated` (mode, seconds).
- Export → `plan_exported` / `export_downloaded` (format).
- First project creation → `project_created` (anonymous/auth).
- Opening an existing project via link → `project_returned`.

**Privacy notice** (About page, plan 10): what is collected (event names, timings, counts, a hash of the topic text), what isn't (IP, raw topics unless enabled, paper text), retention (`retention_days`), and an **opt-out toggle** that sets `telemetry_opt_out`. Add `purge-events` to the persistence CLI and run it in the weekly job (plan 11).

---

## 10. Admin dashboard: `app/views/admin.py` (same gating as Evaluate)

- **Funnel (last 30 days):** sessions with `topic_started` → `reading_viewed` → `gaps_generated` → `gap_saved` → `plan_generated` → `project_returned` within 7 days. Show counts and step conversion.
- **Quality:** the latest gap/extraction/plan metrics from `reports/eval/history.jsonl`, with the launch bars.
- **"Not useful" reasons:** a bar chart, plus the 10 most recent cards flagged (links to them via the stored gap_id → workspace cache).
- **Free quota:** LLM requests per day and per task/route from the usage ledger, today's remaining quota per route (`QuotaManager.remaining_today`), the days on which quota ran out, and the average build times.
- **Data freshness:** the bundle release, generated_at, extraction coverage and full-text coverage.

Charts via Plotly with the existing `ui.apply_chart_style`.

---

## 11. Documentation: `Docs/study_protocol.md` (new)

Write this document with the following sections; it is the plan for non-code validation.

1. **Discovery interviews** (before and during plans 01–03): 8–10 early researchers (final-year undergraduates, first-year master's/PhD students, engineers moving into research). A 30-minute semi-structured guide:
   - How did you find your current or last research topic?
   - What took longest?
   - What did a supervisor reject, and why?
   - Which tools did you use (Scholar, Semantic Scholar, ChatGPT/Claude, Connected Papers…), and where did they fail?
   - What would make you trust a suggested research gap?
   Record anonymized notes; synthesize them into 5 needs ranked by frequency.
2. **Wizard-of-Oz gap cards:** for 3 topics, hand-make 5 cards each with the Gap Card template (statement, evidence quotes, why it might not be a gap, first steps). Show them to students and 2–3 supervisors. Measure: which they'd pursue, which they'd reject and why. **Go/no-go for plan 06:** at least 50% of hand-made cards are rated "worth pursuing" by supervisors.
3. **Benchmark study** (after plan 08):
   - **Design:** within-subject or two groups of ≥ 6 students each, same 2 topics.
   - **Arms:** A uses Google Scholar + a general-purpose LLM assistant; B uses ScholarGrid. 60 minutes each.
   - **Output:** a shortlist of 3 candidate gaps plus a one-paragraph plan.
   - **Blind rating** by 2+ supervisors with the §6 gap rubric plus "Would you supervise this?".
   - **Primary outcome:** share of shortlisted gaps rated real & open & feasible.
   - **Secondary:** time to shortlist, confidence (self-rated), number of verifiable citations, and errors (claims not supported by any paper).
   - **Analysis:** compare proportions with a two-sided test (or a Bayesian beta-binomial estimate given small samples), and report effect sizes and intervals, not only p-values.
   - **Decision:** ScholarGrid must clearly beat arm A on the primary outcome to justify continued investment.
4. **Ethics and consent:** a participant information sheet, voluntary participation, no personal data in telemetry, and the right to withdraw. Check whether your university requires IRB/ethics approval for user studies.
5. **Templates:** a consent form, an interview guide, the rating sheet (mirroring §6) and a results table.

---

## 12. Tests: `tests/test_evaluation.py`

1. Sampling: stratification respects strata floors, the overlap share is ~20%, and the same seed gives the same round.
2. Metrics on a hand-built ratings frame:
   - precision with majority voting,
   - the novelty false-positive rate counts only `novel` items,
   - precision by confidence,
   - Cohen's κ = 1.0 for identical raters and ≈ 0 for random ones,
   - an empty input returns `None` without crashing.
3. Report writes Markdown containing "Launch bar" lines with ✅/❌.
4. Compare harness with FakeLLM handlers per route (monkeypatch `get_llm` to return fakes that differ) → the report has one row per route, with the requests-per-paper and truncation columns present.
5. Telemetry: with `telemetry.enabled=True`, `track("gap_saved", gap_id="g1")` inserts an `Event` (join the thread in tests by monkeypatching `threading.Thread` to run synchronously); unknown event names are ignored; the opt-out flag suppresses events; `store_queries=False` → no raw `q` in props.
6. App: with `SCHOLARGRID_EVAL_MODE=1` and a round file present, the Evaluate page renders the first item **without** any "Confidence", "score" or "novel" text (blindness test); submitting a rating writes a row. Without the env var, the page is not in navigation.

---

## 13. Acceptance checks

```bash
ruff check . && mypy && pytest -q
alembic upgrade head
python -m scholargrid.evaluation round gaps --name pilot --topics "language models" --n 10 --config configs/ci.yaml
SCHOLARGRID_EVAL_MODE=1 streamlit run app/streamlit_app.py      # More → Evaluate → rate 10 items
python -m scholargrid.evaluation report --batch pilot
cat reports/eval/pilot.md
```

**Before launch (plan 11):** run one real gap round (≥ 3 topics, ≥ 60 cards, ≥ 3 domain experts, 20% overlap) and one extraction round (25 papers). Paste both report headlines into `Docs/runbook.md`.

## 14. Pitfalls

- Never show raters the system's own confidence or novelty verdicts; it anchors them.
- Freeze rounds. Rating cards regenerated after a code change mixes versions and makes metrics meaningless.
- Small samples: report counts and intervals (e.g. a Wilson interval for proportions), not only percentages.
- Telemetry is product data, not research data about people. Keep it minimal, and document it in the About page.
