"""About — what ScholarGrid does, how open problems are found, limits, data, privacy and licences."""
from __future__ import annotations

import html

import streamlit as st

import ui

_JOURNEY = [
    (":material/flag:", "Topic", "Describe a topic; see its sub-topics in plain language."),
    (":material/menu_book:", "Read first", "An ordered reading path, from overviews and foundations to the frontier."),
    (":material/explore:", "Open problems", "Problems authors say are still open, with quotes and a novelty check."),
    (":material/route:", "Plan", "A starter research plan for one problem, grounded in its papers."),
    (":material/folder_open:", "My project", "Everything you saved, kept between sessions."),
]

_STEPS = [
    ("Collect", "Recent arXiv computer-science papers, cleaned and de-duplicated."),
    ("Embed", "Each title and abstract becomes a vector ({embedding}). Your searches use the same model."),
    ("Group", "Similar papers are grouped into research areas ({reduce} → {cluster})."),
    ("Name", "Each area is named from its most distinctive keywords."),
    ("Connect", "Related papers are connected by text similarity, with shared-reference signals when available. Browse these connections one paper at a time."),
    ("Check", "Search quality, cluster quality and robustness are validated on every run."),
    ("Search", "Searches show papers containing your exact phrase (acronyms such as SATD are expanded) "
               "first, then papers with a similar meaning. Besides the collection, each search asks "
               "arXiv, OpenAlex and Semantic Scholar live. Papers found that way are marked "
               "“New · found online”; they are not part of the analyzed areas, map or trends "
               "until the next data build."),
]


def render() -> None:
    bundle = ui.get_bundle()
    meta = bundle.meta
    b = meta.get("backends", {})

    ui.page_header("About", "How ScholarGrid works",
                   "From a topic to a research question, with evidence you can check.")

    with ui.panel("what"):
        ui.panel_header("What ScholarGrid does", "5 steps")
        cols = st.columns(5, gap="small")
        for col, (icon, title, text) in zip(cols, _JOURNEY):
            with col:
                st.markdown(f"#### {icon} {title}")
                st.caption(text)

    _how_gaps(bundle)
    _cannot(bundle)

    with ui.panel("how"):
        ui.panel_header("How the collection is built", f"{len(_STEPS)} steps")
        cols = st.columns(3, gap="medium")
        for i, (title, text) in enumerate(_STEPS):
            cols[i % 3].markdown(
                f"<div class='sg-card' style='animation-delay:{i * 50}ms'>"
                f"<div class='sg-eyebrow'>Step {i + 1}</div><div class='t'>{title}</div>"
                f"<div class='s'>{text.format(**{k: b.get(k, '?') for k in ('embedding', 'reduce', 'cluster', 'landscape')})}</div>"
                f"</div>", unsafe_allow_html=True)

    with ui.panel("limits"):
        ui.panel_header("Keep in mind")
        st.markdown("""
- Research areas are **automatic groupings, not an authoritative taxonomy**. Check the papers behind each label.
- **Citations lag.** Very recent papers have few or none yet. A zero count may also reflect missing metadata.
- **More papers ≠ better research.** Activity is not quality.
- **Growth needs history.** Areas are only called growing when the evidence is strong enough;
  everything else shows counts only.
- **Similar wording ≠ compatible science.**
- **Areas, the map and Trends reflect the last data build**, not today. Search also checks the
  live sources; papers found online are kept apart from the analyzed collection, and on hosted
  versions they may be cleared when the app restarts.
""")

    with ui.panel("dataset"):
        snap = meta.get("snapshot", {})
        counts = meta["counts"]
        release = _release_id(bundle)
        ui.panel_header("Dataset", f"release {release}" if release else snap.get("source", ""))
        papers = max(1, counts["papers"])
        ui.stat_tiles([
            (f"{counts['papers']:,}", "Papers"),
            (counts["clusters"], "Areas"),
            (ui.fmt_date(snap.get("earliest_included_date")), "From"),
            (ui.fmt_date(snap.get("latest_included_date")), "To"),
        ])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        span = snap.get("span_months")
        ui.stat_tiles([
            (f"{span:g} mo" if isinstance(span, (int, float)) else "—", "Time span"),
            (ui.fmt_date(meta.get("generated_at")), "Last refresh"),
            (f"{counts.get('with_citations', 0) / papers:.0%}", "With citations"),
            (f"{counts.get('with_references', 0) / papers:.0%}", "With references"),
        ])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        ft = meta.get("fulltext") or {}
        xb = meta.get("extraction_bundle") or {}
        n_cards = sum(len(v) for v in ((bundle.gaps or {}).get("areas") or {}).values())
        ui.stat_tiles([
            (f"{float(ft.get('coverage') or 0):.0%}" if ft.get("in_scope") else "—", "Full text (in scope)"),
            (f"{int(xb.get('papers_with_extraction') or 0):,}", "Papers extracted"),
            (f"{n_cards:,}", "Area open-problem cards"),
            (release or "—", "Release"),
        ])
        sampled = snap.get("n_available") and snap.get("n_available") != snap.get("n_papers")
        st.caption(
            f"Source: {snap.get('source', '?')}"
            + (f" · profile: {snap.get('profile')}" if snap.get("profile") else "")
            + (f" · stratified sample of {snap['n_available']:,} available papers" if sampled else "")
            + f" · embedding model: {meta.get('embedding_model') or b.get('embedding', '?')}")
        with st.expander("Full run metadata"):
            st.json(meta, expanded=False)

    _grow_panel(bundle)
    _privacy()

    with ui.panel("licence"):
        ui.panel_header("Credits and licences")
        st.markdown("""
- **arXiv**: paper metadata via arXiv's open interfaces. Abstracts keep each paper's own licence;
  every result links back to arXiv, and no PDFs are redistributed. *Thank you to arXiv for use of
  its open access interoperability.*
- **OpenAlex**: search results, citation counts, references and venues (CC0).
- **Semantic Scholar**: search results via the Semantic Scholar Academic Graph API.
- Abstracts and metadata from every source link back to the original record.
- **Quotes are short excerpts with links to the source; each paper keeps its own licence.**
  Full texts are read for analysis only and never redistributed.
""")


def _how_gaps(bundle) -> None:
    from scholargrid.gaps.model import TYPE_HELP, TYPE_LABELS

    with ui.panel("how-gaps"):
        ui.panel_header("How open problems are found")
        st.markdown("\n".join(f"- **{TYPE_LABELS[t]}.** {TYPE_HELP[t]}" for t in TYPE_LABELS))
        st.markdown(
            "**Quotes are verified.** Every quote on a card was found, character for character, in the paper's "
            "text (or it is not shown). Paraphrased results are labelled as paraphrases.\n\n"
            "**Novelty is checked.** Each card is compared with recent papers in the collection and, for topic "
            "cards, with live arXiv, OpenAlex and Semantic Scholar searches. The card says what was checked and "
            "when. “Novel” means no match was found among what was checked, not that none exists.\n\n"
            "**Scores** combine five parts, each explained on the card: evidence (how many papers support it), "
            "novelty, feasibility (data, code, compute), specificity, and recency of the evidence. Confidence is "
            "high only with strong evidence and a live novelty check.")
        st.markdown("**Why the old “Investigation leads” were replaced.** They found isolated papers in the "
                    "embedding space, which rarely correspond to research gaps. Open problems use what authors "
                    "state, structured facts from the papers, and novelty checks instead.")


def _cannot(bundle) -> None:
    scope = ui.get_config()["scope"]["name"]
    with ui.panel("cannot"):
        ui.panel_header("What it cannot do")
        st.markdown(f"""
- It **does not judge scientific value**. A card is a lead to check, not a verdict.
- It covers **{html.escape(str(scope)) if scope != 'all' else 'the collected'} arXiv papers** plus live sources.
  Work outside them (other venues, industry, different terminology) can be missed by the novelty check.
- **AI-written text is grounded but can still be wrong.** Always read the evidence and the papers.
- Areas are automatic groupings, citations lag for recent papers, and growth needs history.
""")


def _privacy() -> None:
    cfg = ui.get_config()
    t, p = cfg["telemetry"], cfg["persistence"]
    with ui.panel("privacy"):
        ui.panel_header("Privacy")
        st.markdown(f"""
- **Usage statistics:** we record anonymous event names (for example “opened a topic”), counts, timings and a
  hash of the topic text, so we can see which steps help. We do **not** record IP addresses, paper text, or the
  topic text itself{" (unless an administrator turned that on)" if t["store_queries"] else ""}. Events are kept
  for {int(t["retention_days"])} days.
- **Projects** are stored {"behind an unguessable link (anyone with the link can open it)" if p["auth"] != "oidc"
  else "in your account when you sign in, or behind an unguessable link"}. Delete a project any time from
  **My project → Settings**.
- **AI requests** go to free AI providers (Google Gemini's free tier). We send only public paper text and topic
  queries; free-tier content may be used by the provider to improve its products. The background you type on the
  Plan page is sent too, so leave out anything personal.
""")
        st.toggle("Don't record anonymous usage statistics for this session", key="telemetry_opt_out")


def _start_grow() -> None:
    from scholargrid.grow import start

    start(ui.get_config())


def _grow_panel(bundle) -> None:
    from scholargrid.grow import log_tail, read_status

    cfg = ui.get_config()
    if not cfg["app"]["allow_grow"]:
        return
    status = read_status(cfg)
    state = status.get("state", "idle")
    target = int(cfg.max_papers or 0)
    with ui.panel("grow"):
        ui.panel_header("Grow the collection", f"{len(bundle.df):,} papers now")
        st.write(f"Collect up to {target:,} arXiv papers month by month across the whole date window. "
                 "This runs in the background (roughly 30–60 minutes) while search keeps working; "
                 "the larger collection is picked up automatically when it is ready. "
                 "Search already checks arXiv, OpenAlex and Semantic Scholar live, so this mainly "
                 "improves Areas, Trends and the research map.")
        if state == "running":
            st.progress(float(status.get("progress") or 0.0), text=status.get("message") or "Working…")
            st.caption(f"Started {ui.fmt_date(status.get('started_at'))}. Reload this page for an update.")
            st.button("Refresh status", key="grow_refresh")
            return
        if state == "done":
            ui.note(f"<b>Last run finished</b> {html.escape(ui.fmt_date(status.get('finished_at')))}"
                    + (f": release {html.escape(str(status['release']))}" if status.get("release") else "")
                    + (f" with {int(status['papers']):,} papers" if status.get("papers") else "") + ".")
        elif state == "failed":
            ui.note(f"<b>The last run failed.</b> {html.escape(str(status.get('message') or ''))} "
                    "Finished months are cached, so trying again continues where it stopped.")
            with st.expander("Log"):
                st.code(log_tail(cfg) or "(empty)")
        st.button("Try again" if state == "failed" else "Grow collection", type="primary",
                  key="grow_start", on_click=_start_grow)


def _release_id(bundle) -> str | None:
    import os

    manifest = os.path.join(bundle.path or "", "manifest.json")
    if os.path.exists(manifest):
        from scholargrid.utils import load_json

        return load_json(manifest).get("release")
    return None
