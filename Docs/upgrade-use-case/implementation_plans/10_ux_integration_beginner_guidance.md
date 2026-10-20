# Plan 10: UX integration and beginner guidance

> **Prerequisites:** plans 05–09 are merged (journey pages, Gap Cards, plans, projects, telemetry).
> **Produces:** a coherent beginner journey across all pages: a journey bar and "next step" calls to action; the Trends page reframed as an **opportunity quadrant** (growth × crowdedness × open problems) with plain advice; the old Leads retired from navigation; a glossary with inline term tooltips; beginner explainers (how to read a paper, how to judge a gap, what makes a good first project); consistent empty states; an About/methodology rewrite with a privacy section; a **demo mode** that runs the whole journey offline; and accessibility and mobile fixes.
> **Why:** the parts exist after plans 05–09; this plan makes them feel like one product a newcomer can follow without help.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 10_ux_integration_beginner_guidance.md. Do not change analytics or LLM logic;
this is presentation, copy, navigation and demo wiring. Keep every page test in tests/test_app.py
passing and add the new ones listed in §11. Order: §3 shared UI components → §4 journey bar + next
steps → §5 opportunity quadrant → §6 retire Leads → §7 glossary + explainers → §8 About rewrite →
§9 demo mode → §10 accessibility/mobile → §11 tests. Finish with §12.
```

---

## 2. Principles for all copy

- Speak to a smart newcomer: short sentences, define jargon once (the glossary, §7), no hype.
- Every page answers three questions at the top: **where am I, what can I do here, what's next**.
- Honesty is structured, not repeated: each Gap Card has its own "why this might not be a gap" and novelty line. Remove generic page-wide "not a proven gap" banners from the journey pages.
- Numbers always come with their counts and basis ("based on 412 papers retrieved for your topic").

---

## 3. Shared UI components (`app/ui.py`)

```python
def empty_state(title: str, body: str, actions: Sequence[tuple[str, str]] = ()) -> None:
    """Consistent empty/error state. actions = [(button label, page key), ...]."""
    with panel(f"empty-{abs(hash(title)) % 10**8}"):
        st.markdown(f"<div class='sg-empty'><div class='t'>{html.escape(title)}</div>"
                    f"<p>{html.escape(body)}</p></div>", unsafe_allow_html=True)
        cols = st.columns(max(1, len(actions)))
        for col, (label, key) in zip(cols, actions):
            if col.button(label, key=f"empty_{key}_{abs(hash(title)) % 10**6}"):
                goto(key)


def next_step(label: str, page_key: str, hint: str = "") -> None:
    """Big call-to-action at the bottom of a journey page."""
    with panel(f"next-{page_key}"):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown(f"**Next:** {html.escape(label)}" + (f"<br><span class='sg-muted'>{html.escape(hint)}</span>"
                    if hint else ""), unsafe_allow_html=True)
        if c2.button("Continue →", type="primary", key=f"next_{page_key}", use_container_width=True):
            goto(page_key)


def chip(text: str, kind: str = "") -> str:
    return f"<span class='sg-chip {html.escape(kind)}'>{html.escape(text)}</span>"
```

Add CSS for `.sg-empty`, `.sg-muted`, `.sg-journey-bar` (a step list with `.done` / `.current` states), and confidence/feasibility chip colours. Keep contrast ≥ 4.5:1 and never rely on colour alone: include the word ("High confidence").

---

## 4. Journey bar and next steps

`ui.journey_bar(current)` (introduced in plan 05) appears at the top of `topic`, `reading`, `gaps`, `plan` and `project`. Each page ends with `ui.next_step(...)`:

| Page | Next step label | Target |
|---|---|---|
| Topic | "See what to read first" | `reading` |
| Read first | "Find open problems in this topic" | `gaps` |
| Open problems | "Turn a problem you like into a starter plan" (only enabled after saving or choosing one) | `plan` |
| Plan | "Save it to your project and share it with a supervisor" | `project` |
| My project | "Explore another topic" | `start` |

The Start page shows a three-card preview of what the user will get: a sub-topic card, a reading-path item and a Gap Card. Use a **static example** from `app/assets/example_gap.json`, clearly labelled "Example".

---

## 5. Trends → Opportunity view (`app/views/trends.py`)

Keep the existing growth tables (they're good) as a second tab, **"Growth details"**. Add a first tab, **"Where to look"**:

### 5.1 Data (`scholargrid/opportunity.py`, pure function, unit-tested)

```python
def opportunity_table(bundle) -> pd.DataFrame:
    """One row per area: name, size, recent_papers (last 12 months), crowdedness (recent_papers percentile
    0..1), growth (relative growth if reliable else NaN), growth_ci, reliable, open_problems (number of
    gap cards for the area in bundle.gaps), stated_per_100 (verified stated problems per 100 extracted
    papers), quadrant."""
```

Quadrant rule:
- `crowded = crowdedness ≥ 0.5`
- `growing = reliable and ci_low > 1`

| | less crowded | crowded |
|---|---|---|
| **growing** | **Emerging**: "Growing and not yet crowded. Good entry point if the problems are feasible." | **Hot & competitive**: "Many groups publish here. Pick a sharp niche, like an evaluation gap or a specific setting." |
| **not growing / no reliable signal** | **Niche**: "Fewer papers. Check that the problems matter (citations, venues) before committing." | **Mature**: "Established area. Reproductions, evaluation gaps and unsettled questions are realistic first projects." |

### 5.2 Chart
Plotly scatter: x = crowdedness percentile, y = growth (reliable only; unreliable areas plotted at y = 1.0 in grey with an open marker and the hover note "no reliable growth signal"), size = `open_problems + 1`, hover = name, recent papers, growth with CI, open problems. Dotted lines at x = 0.5 and y = 1.0, and quadrant labels in the corners. Below the chart, `st.caption` with a text summary of the chart (accessibility). Then a table sorted by open problems, with a button per row: "Open problems in this area" (sets the area and goes to `gaps` in area mode).

### 5.3 Workspace variant
On the Topic page, add a small version for sub-topics: x = size percentile within the workspace, y = `recent_share`, size = `n_open_problems`. Label it **"Based on the papers retrieved for your topic; not a growth estimate"**.

---

## 6. Retire the old Leads

- Add `app.show_legacy_leads: bool = False` to the schema and config. When false, `_build_pages` doesn't register `leads`.
- Keep `scholargrid/sparse.py` and the `sparse` step in `analyze` (it is cheap and keeps old bundles comparable), but add `sparse.enabled: bool = True` to allow skipping it.
- About page: one paragraph, "Why 'Investigation leads' were replaced": they found isolated papers in embedding space, which rarely correspond to research gaps. The Open problems page uses what authors state, structured facts, and novelty checks instead.
- `tests/test_app.py`: keep `leads` in `PAGES` only if the flag is true. Add a test that the nav doesn't contain "Leads" by default.

---

## 7. Glossary and beginner explainers

### 7.1 `app/glossary.py`

```python
GLOSSARY = {
    "research gap": "A question the field has not answered yet that is worth answering.",
    "open problem": "A problem that authors explicitly say is still unsolved.",
    "survey": "A paper that summarises a whole area. The best first read.",
    "baseline": "An existing method you compare your idea against.",
    "benchmark": "A standard dataset and task used to compare methods.",
    "ablation": "An experiment that removes one part of a method to see what it contributes.",
    "state of the art (SOTA)": "The best reported result on a benchmark at a given time.",
    "reproduction": "Re-running a paper's experiments to check its results; a good first project.",
    "novelty check": "Our search of recent papers for work that already does what a gap suggests.",
    "co-citation": "Two papers cited together by many others; we use it to find foundations.",
    "sub-topic": "A group of closely related papers within your topic.",
    "evaluation gap": "A method that has only been tested in narrow conditions.",
    "citation": "A reference from one paper to another; counts lag for recent papers.",
    "preprint": "A paper shared publicly (e.g. on arXiv) before or without peer review.",
}


def term(word: str) -> str:
    """Inline tooltip: <abbr title='...'>word</abbr> (screen readers read the title)."""
```

Use `term()` in the page leads (e.g. on Read first: "Start with a {survey}, then the {co-citation} foundations…"). Add a **"New to research? Key terms"** popover in the top nav that lists the glossary.

### 7.2 Explainers (collapsed expanders, short, practical)
- **Read first → "How to read a research paper"**: the three-pass method (skim title/abstract/figures/conclusion; read for the main idea; read in depth when reproducing), plus "what to note for each paper" (problem, method, data, results, limitations), and a link to the project notes.
- **Open problems → "How to judge a research gap"**: is it real (do several papers mention it)? Is it still open (check the novelty line, and search the last 6 months yourself)? Does it matter (who would use the result)? Can you do it (data, code, compute, time)? Would a supervisor agree (questions to ask)?
- **Plan → "What makes a good first project"**: small and concrete; start by reproducing a baseline; one clear change; a defined evaluation; a minimum viable result you can write up.

---

## 8. About page rewrite (`app/views/about.py`)

Sections:
1. **What ScholarGrid does**: the journey in 5 steps with icons.
2. **How open problems are found**: the six gap types with one-line explanations (from `TYPE_HELP`), how quotes are verified, how novelty is checked, and what the scores mean.
3. **What it cannot do**: it doesn't judge scientific value; it covers {scope} papers from arXiv plus live sources; the novelty check can miss work outside those sources or under different names; LLM-written text is grounded but can still be wrong, so always read the evidence.
4. **Data and freshness**: keep the existing dataset tiles; add full-text coverage, extraction coverage, gap card counts and the release id.
5. **Privacy**: what telemetry collects (plan 09), the opt-out toggle, how projects are stored (link-based or account), and how to delete a project.
6. **Credits and licences**: the existing attribution footer text, plus "Quotes are short excerpts with links to the source; each paper keeps its own licence."

---

## 9. Demo mode (offline, whole journey)

`views/onboarding.py::_build_demo` currently builds a synthetic bundle. Extend it so every new stage produces content offline:

```python
    demo.raw["data"]["source"] = "synthetic"
    demo.raw["data"]["max_papers"] = 800
    demo.raw["llm"]["provider"] = "fake"            # deterministic, no key, no cost
    demo.raw["fulltext"]["source"] = "synthetic"
    demo.raw["search"]["live"] = False
    demo.raw["persistence"]["default_url"] = "sqlite:///data/demo_app.db"
```

Requirements:
- The demo build runs in < 60 s on a laptop (tfidf/pca backends are fine).
- A global banner (existing synthetic-data notice) adds: "Demo: papers, problems and plans are generated from synthetic data to show how the tool works."
- Start page example chips in demo mode use topics that exist in the synthetic corpus ("language models", "federated learning privacy", "motion planning").

---

## 10. Accessibility and mobile

- Every interactive element has a visible label; icon-only buttons get `help=` text.
- Headings: one `h1` per page (`page_header`), then panels as `h2`-level visual headers. Don't skip levels in custom HTML.
- Charts: each has a text caption that summarizes it. Tables are not images.
- Focus: Streamlit handles native widgets. For custom HTML links, keep the default focus outline (don't remove it in CSS).
- Mobile (< 640 px): add CSS media queries so 3-column card grids become 1 column, the journey bar wraps, and Gap Card chips wrap. Test in browser dev tools at 375 px.
- Reduced motion: the existing `prefers-reduced-motion` rules must also cover the new card animations.

Browser end-to-end tests (Playwright, which downloads a Chromium browser) are **deferred** (`LATER_NOT_NOW.md`).
For now: the Streamlit `AppTest` page tests below (they run in CI), plus a manual walkthrough of the whole
journey on the deployed Hugging Face Space and a 375 px check in the browser's dev tools.

---

## 11. Tests

1. `tests/test_opportunity.py`: quadrant assignment on a hand-built table (reliable growth > 1 & crowded → "Hot & competitive", etc.). Unreliable growth never yields "growing".
2. `tests/test_app.py`:
   - nav has no "Leads" by default,
   - the Trends page renders both tabs and the caption text,
   - each journey page shows the journey bar and its "Next:" call to action (with a demo workspace in the session),
   - the About page contains "How open problems are found" and "Privacy",
   - `glossary.term("baseline")` returns an `<abbr` element with the definition.
3. `tests/test_demo.py`: run the demo build config through `pipeline.run` → the bundle has reading paths and ≥ 1 gap card; the Start page in demo mode lists the demo example topics.

---

## 12. Acceptance checks

```bash
ruff check . && mypy && pytest -q
rm -rf data/processed data/releases && streamlit run app/streamlit_app.py   # "Load demo data" → full journey offline
```

Manual walkthrough with **one person who has never seen the app** (a classmate is fine). Ask them to go from "I'm interested in X" to a saved plan without help, while you take notes silently. Fix the three biggest stumbling points before plan 11. Record what you changed in `Docs/ux-review.md` under a new dated section.
