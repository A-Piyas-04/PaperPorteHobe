"""Shared UI layer for the ScholarGrid app.

Global light-editorial stylesheet, loading/transition effects, reusable
components (panels, cards, pills, tables, skeletons) and the data helpers every
page needs.

Design rules: a 16px type scale with clear heading steps, white section
panels on a soft grey page so sections are clearly separated, flat bordered
cards with a small hover lift, fade/slide entrances, a thin top progress bar
while the app is working. No dark/glass surfaces, gradient heroes, glows,
spotlights or ring effects.
"""
from __future__ import annotations

import contextlib
import html
import os
import sys
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.artifacts import load_bundle  # noqa: E402
from scholargrid.config import load_config  # noqa: E402
from scholargrid.labeling import pretty_label  # noqa: E402

# --- palette ---------------------------------------------------------------
ACCENT = "#2f6feb"
INK = "#161a22"
MUTED = "#566070"
BORDER = "#e2e5ea"
BG = "#f4f5f7"
NOISE_COLOR = "#cfd3da"
PALETTE = [
    "#2f6feb", "#e8710a", "#1a9e6c", "#d1453b", "#7b57d6", "#c99700",
    "#0b8aa6", "#c2457f", "#6b8e23", "#8a6d3b", "#4b6bdb", "#2aa198",
    "#b5532a", "#5f6bd8", "#3f8f3f", "#a0469a", "#0f7c8c", "#b0761b",
    "#6a5acd", "#2e7d5b",
]
FONT = "'Source Sans Pro', 'Source Sans 3', -apple-system, 'Segoe UI', sans-serif"

EXAMPLES = [
    "large language model agents",
    "diffusion models for images",
    "graph neural networks",
    "federated learning privacy",
    "reinforcement learning for robotics",
    "efficient transformer inference",
]


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
_CSS = """
<style>
  :root {
    --accent:#2f6feb; --ink:#161a22; --muted:#566070; --border:#e2e5ea;
    --bg:#f4f5f7; --soft:#eef2fb;
  }

  /* ---- Type scale (16px base): 40 / 22 / 18 / 16 / 14 / 13 ----------- */
  html { font-size: 16px !important; }
  .stApp { background: var(--bg); color: var(--ink); }
  .block-container {
    max-width: 1240px; padding-top: 0 !important; padding-bottom: 4rem;
  }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li { font-size: 1rem; line-height: 1.6; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    font-size: .875rem !important; color: var(--muted) !important;
  }
  [data-testid="stWidgetLabel"] p {
    font-size: .875rem !important; font-weight: 700; color: var(--ink);
  }
  [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"],
  [data-testid="stSliderTickBarMax"] {
    font-family: inherit !important; font-size: .875rem !important;
  }
  [data-baseweb="select"] * , [data-baseweb="tag"] span { font-size: .9375rem !important; }
  a { color: var(--accent); }

  /* ---- Chrome: hide dev header/status, keep a clean canvas ------------ */
  [data-testid="stHeader"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="stHeadingWithActionElements"] a,
  [data-testid="stHeaderActionElements"] { display: none !important; }

  /* ---- Top progress bar while a rerun / page switch is running -------- */
  .stApp::before {
    content: ""; position: fixed; top: 0; left: 0; height: 3px; width: 35vw;
    background: var(--accent); z-index: 1000000; border-radius: 0 3px 3px 0;
    opacity: 0; transform: translateX(-110%); transition: opacity .25s ease;
    pointer-events: none;
  }
  .stApp[data-test-script-state="running"]::before {
    opacity: 1; animation: sgBar 1.1s cubic-bezier(.4,0,.2,1) infinite;
  }
  @keyframes sgBar { from { transform: translateX(-110%); } to { transform: translateX(300%); } }
  /* Keep content steady during reruns instead of Streamlit's heavy fade. */
  [data-testid="stElementContainer"][data-stale="true"] {
    opacity: .85 !important; transition: opacity .3s ease .25s;
  }

  /* ---- Entrance animations ------------------------------------------- */
  @keyframes sgRise { from { opacity: 0; transform: translateY(14px); }
                      to   { opacity: 1; transform: none; } }
  @keyframes sgFade { from { opacity: 0; } to { opacity: 1; } }
  @keyframes sgShimmer { to { transform: translateX(100%); } }
  @keyframes sgSlide { from { transform: translateX(-100%); } to { transform: translateX(250%); } }

  /* ---- Top navigation ------------------------------------------------- */
  .st-key-topnav {
    position: sticky; top: 0; z-index: 999; background: var(--bg);
    padding: .85rem 0 .75rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  .sg-brand { font-size: 1.375rem; font-weight: 800; color: var(--ink);
              white-space: nowrap; letter-spacing: -.01em; }
  .st-key-topnav [data-testid="stPageLink"] a {
    padding: .5rem .9rem; border-radius: 10px; justify-content: center;
    transition: background .18s ease, color .18s ease;
  }
  .st-key-topnav [data-testid="stPageLink"] a p {
    font-size: 1rem !important; font-weight: 650; color: var(--muted);
  }
  .st-key-topnav [data-testid="stPageLink"] a:hover { background: #e8ecf5; }
  .st-key-topnav [data-testid="stPageLink"] a:hover p { color: var(--ink); }
  .st-key-topnav [data-testid="stPageLink"] a[aria-current="page"],
  .st-key-topnav [data-testid="stPageLink"] a[aria-current="true"] { background: var(--soft); }
  .st-key-topnav [data-testid="stPageLink"] a[aria-current="page"] p,
  .st-key-topnav [data-testid="stPageLink"] a[aria-current="true"] p { color: var(--accent); }

  /* ---- Section panels: white blocks on the grey page ----------------- */
  div[class*="st-key-panel-"] {
    background: #fff; border: 1px solid var(--border); border-radius: 16px;
    padding: 1.75rem 2rem 1.6rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(16,24,40,.05);
    animation: sgRise .5s cubic-bezier(.2,.7,.2,1) both;
  }
  .sg-ph { display: flex; align-items: baseline; justify-content: space-between;
           gap: 1rem; padding-bottom: .85rem; margin-bottom: 1.15rem;
           border-bottom: 1px solid #eceef2; }
  .sg-ph-t { font-size: 1.375rem; font-weight: 800; color: var(--ink);
             letter-spacing: -.01em; }
  .sg-ph-m { font-size: .875rem; color: var(--muted); font-weight: 600;
             white-space: nowrap; }

  /* ---- Page header ---------------------------------------------------- */
  .sg-eyebrow { color: var(--accent); font-weight: 800; letter-spacing: .1em;
                text-transform: uppercase; font-size: .8125rem; margin-bottom: .5rem; }
  .sg-title { font-size: 2.5rem; font-weight: 800; line-height: 1.12;
              color: var(--ink); letter-spacing: -.02em; margin: 0 0 .6rem; }
  .sg-lead { color: var(--muted); font-size: 1.125rem; line-height: 1.55;
             margin: 0 0 1.4rem; max-width: 46rem; }
  .sg-page-head { animation: sgRise .45s cubic-bezier(.2,.7,.2,1) both;
                  margin: .4rem 0 1.6rem; }

  /* ---- Search input --------------------------------------------------- */
  [data-testid="stTextInput"] [data-baseweb="input"] {
    border-radius: 16px !important; border: 1.5px solid #d5d9e0 !important;
    background: #fff !important; transition: border-color .18s ease, box-shadow .18s ease;
  }
  [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 4px #e3ebfc;
  }
  [data-testid="stTextInput"] input {
    height: 3.4rem; font-size: 1.125rem !important; padding: 0 1.1rem;
    background: #fff !important; color: var(--ink);
  }

  /* ---- Buttons -------------------------------------------------------- */
  [data-testid="stButton"] button { transition: all .18s ease; }
  [data-testid="stButton"] button p { font-size: .9375rem !important; font-weight: 650; }
  [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    min-height: 2.75rem; border-radius: 10px !important;
  }
  [data-testid="stBaseButton-secondary"] {
    border: 1.5px solid var(--border) !important; background: #fff !important;
  }
  [data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent) !important; transform: translateY(-1px);
    background: #fbfcff !important;
  }
  [data-testid="stBaseButton-secondary"]:hover p { color: var(--accent) !important; }
  [data-testid="stButton"] button:active { transform: translateY(0); }

  [data-testid="stBaseButton-primary"], [data-testid="stButton"] button[kind="primary"] {
    background: var(--accent) !important; border-color: var(--accent) !important;
    color: #fff !important;
  }
  [data-testid="stBaseButton-primary"] p { color: #fff !important; }
  [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.06); }
  [data-testid="stBaseButton-tertiary"] p { color: var(--accent) !important; }

  /* ---- Stat tiles ----------------------------------------------------- */
  .sg-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 1rem; }
  .sg-tile { background: #f8f9fb; border: 1px solid #eceef2; border-radius: 16px;
             padding: 1.3rem 1.4rem; }
  .sg-tile .v { font-size: 2rem; font-weight: 800; color: var(--ink); line-height: 1.05; }
  .sg-tile .k { font-size: .8125rem; color: var(--muted); margin-top: .45rem;
                text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }

  /* ---- Paper card ----------------------------------------------------- */
  .sg-card {
    background: #fff; border: 1px solid var(--border); border-radius: 16px;
    padding: 1.2rem 1.4rem; margin-bottom: .9rem;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    animation: sgRise .45s cubic-bezier(.2,.7,.2,1) both;
  }
  .sg-card:hover { transform: translateY(-2px); border-color: #cfd5de;
                   box-shadow: 0 8px 24px rgba(16,24,40,.08); }
  .sg-card .t { font-size: 1.125rem; font-weight: 700; line-height: 1.4; color: var(--ink); }
  .sg-card .t a { color: var(--ink); text-decoration: none; }
  .sg-card .t a:hover { color: var(--accent); }
  .sg-card .a { color: var(--muted); font-size: .9375rem; margin-top: .3rem; }
  .sg-card .m { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .6rem;
                align-items: center; }
  .sg-card .s { color: #3d4553; font-size: .9375rem; margin-top: .65rem; line-height: 1.6; }
  .sg-chip { background: #f1f3f6; color: #3b4352; border-radius: 8px;
             padding: .18rem .6rem; font-size: .8125rem; font-weight: 650; }
  .sg-chip.area { background: var(--soft); color: #1f4fb8; }
  .sg-chip.cites { background: #fdf3e6; color: #8a4b08; }
  .sg-score { margin-left: auto; color: var(--muted); font-size: .8125rem; font-weight: 650; }
  .sg-card mark { background: #fff1c2; color: inherit; padding: 0 .1rem; border-radius: 3px; }
  a:focus-visible, button:focus-visible { outline: 3px solid #9ab8f5 !important; outline-offset: 2px; }

  /* ---- Area card (keyed container) ----------------------------------- */
  div[class*="st-key-card-"] {
    background: #fff; border: 1px solid var(--border); border-radius: 18px;
    padding: 1.4rem 1.5rem 1.2rem; height: 100%;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    animation: sgRise .5s cubic-bezier(.2,.7,.2,1) both;
  }
  div[class*="st-key-card-"]:hover { transform: translateY(-3px); border-color: #cfd5de;
                                     box-shadow: 0 10px 28px rgba(16,24,40,.09); }
  .sg-area-t { font-size: 1.125rem; font-weight: 800; color: var(--ink); line-height: 1.3; }
  .sg-area-m { color: var(--muted); font-size: .875rem; font-weight: 600; margin-top: .35rem; }
  .sg-swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 4px;
               margin-right: .5rem; vertical-align: middle; }

  /* ---- Pills ---------------------------------------------------------- */
  .sg-pills { display: flex; flex-wrap: wrap; gap: .45rem; margin: .8rem 0 .6rem; }
  .sg-pill { background: #f1f3f6; color: #333b49; border-radius: 999px;
             padding: .28rem .8rem; font-size: .8125rem; font-weight: 650; }
  .sg-badge { display: inline-block; padding: .2rem .7rem; border-radius: 999px;
              font-size: .8125rem; font-weight: 750; margin-left: .4rem; }
  .sg-up { background: #e5f4ec; color: #17734c; }
  .sg-flat { background: #eef0f3; color: #4d5665; }

  /* ---- Callout -------------------------------------------------------- */
  .sg-note { background: #f3f6fd; border: 1px solid #dce5f8; border-radius: 14px;
             padding: .95rem 1.15rem; color: #27324a; font-size: .9375rem; margin: .2rem 0 1rem;
             animation: sgFade .4s ease both; }
  .sg-note b { color: var(--ink); }

  /* ---- Meter ---------------------------------------------------------- */
  .sg-meter { background: #eceef2; border-radius: 999px; height: .6rem; overflow: hidden;
              margin: .5rem 0 .2rem; }
  .sg-meter > span { display: block; height: 100%; background: var(--accent);
                     border-radius: 999px; transform-origin: left;
                     animation: sgGrow .8s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes sgGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

  /* ---- Table ---------------------------------------------------------- */
  .sg-table { width: 100%; border-collapse: collapse; font-size: .9375rem; }
  .sg-table th { text-align: left; color: var(--muted); font-weight: 800; font-size: .75rem;
                 text-transform: uppercase; letter-spacing: .06em; padding: .7rem .8rem;
                 border-bottom: 2px solid var(--border); }
  .sg-table td { padding: .9rem .8rem; border-bottom: 1px solid #eef0f3; color: var(--ink); }
  .sg-table tr { transition: background .15s ease; }
  .sg-table tbody tr:hover td { background: #f8f9fb; }
  .sg-table td.num, .sg-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .sg-bar { display: inline-block; height: .55rem; border-radius: 999px;
            background: var(--accent); vertical-align: middle; margin-right: .6rem;
            transform-origin: left; animation: sgGrow .7s cubic-bezier(.2,.7,.2,1) both; }

  /* ---- Skeleton loaders ---------------------------------------------- */
  .sg-skel { background: #fff; border: 1px solid var(--border); border-radius: 16px;
             padding: 1.3rem 1.4rem; margin-bottom: .9rem; }
  .sg-skel i { display: block; height: .95rem; border-radius: 6px; background: #eceef2;
               margin: .6rem 0; position: relative; overflow: hidden; }
  .sg-skel i::after { content: ""; position: absolute; inset: 0; transform: translateX(-100%);
                      background: linear-gradient(90deg, transparent, rgba(255,255,255,.75), transparent);
                      animation: sgShimmer 1.2s infinite; }
  .sg-skel-block { height: 560px; border-radius: 16px; background: #f1f3f6;
                   position: relative; overflow: hidden; }
  .sg-skel-block::after { content: ""; position: absolute; inset: 0; transform: translateX(-100%);
                          background: linear-gradient(90deg, transparent, rgba(255,255,255,.6), transparent);
                          animation: sgShimmer 1.4s infinite; }

  /* ---- Splash --------------------------------------------------------- */
  .sg-splash { position: fixed; inset: 0; z-index: 999990; background: var(--bg);
               display: flex; flex-direction: column; align-items: center;
               justify-content: center; gap: 1rem; animation: sgFade .3s ease both; }
  .sg-splash .logo { font-size: 3.4rem; line-height: 1; }
  .sg-splash .name { font-size: 2.2rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; }
  .sg-splash .sub { color: var(--muted); font-size: 1.05rem; }
  .sg-splash .bar { width: 280px; height: 6px; background: #e1e4ea; border-radius: 999px;
                    overflow: hidden; margin-top: .6rem; }
  .sg-splash .bar span { display: block; width: 40%; height: 100%; background: var(--accent);
                         border-radius: 999px; animation: sgSlide 1.1s ease-in-out infinite; }

  .sg-brand-sub { margin-left: 1rem; font-weight: 400; font-size: .9rem; color: var(--muted); }
  .sg-journey { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin: 1rem 0 2rem; }
  .sg-journey > div { border-top: 2px solid #c4d5ee; padding-top: 16px; }
  .sg-journey p { color: var(--muted); margin-top: 8px; }
  .sg-title { max-width: 850px; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.15; }
  .sg-card:hover { transform: none; box-shadow: none; }
  .st-key-panel-hero { padding-top: 2.5rem; }
  @media (max-width: 700px) {
    .sg-journey { grid-template-columns: 1fr; gap: 12px; }
    .sg-brand-sub { display: none; }
    div[class*="st-key-panel-"] { padding: 1.25rem; }
    .st-key-topnav { position: static; }
    .st-key-topnav [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .25rem !important; }
    .st-key-topnav [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
    .st-key-topnav [data-testid="stPageLink"] a { padding: .5rem .2rem; }
    .st-key-topnav [data-testid="stPageLink"] a p { font-size: .8rem !important; }
    .st-key-topnav [data-testid="stIconMaterial"] { display: none; }
  }
</style>
"""

_SPLASH = """
<div class="sg-splash">
  <div class="logo">🔭</div>
  <div class="name">ScholarGrid</div>
  <div class="sub">Loading the research landscape…</div>
  <div class="bar"><span></span></div>
</div>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_bundle():
    bundle = load_bundle(get_config())
    if get_config()["app"]["pre_warm"] and bundle.embedder is not None:
        with contextlib.suppress(Exception):
            bundle.embedder.encode_query("warm up")
    return bundle


def ensure_loaded() -> None:
    """Show a branded full-page splash while the bundle loads (once per session)."""
    if st.session_state.get("_loaded"):
        return
    splash = st.empty()
    splash.markdown(_SPLASH, unsafe_allow_html=True)
    get_bundle()
    st.session_state["_loaded"] = True
    splash.empty()


def bundle_version() -> str:
    return str(get_bundle().meta.get("generated_at", "v1"))


@st.cache_data(show_spinner=False, max_entries=512)
def run_search(version: str, query: str, top_k: int, filters: tuple = (), sort: str = "relevance",
               citation_weight: float = 0.0) -> dict:
    """Cached search keyed by bundle version + query + filters."""
    from scholargrid.search import SearchFilters, search

    b = get_bundle()
    f = SearchFilters(*filters) if filters else None
    return search(query, b.embedder, b.embeddings, b.df, b.labels, b.clusters_meta, top_k=top_k,
                  bm25=b.bm25, index=b.index, cfg=get_config(), filters=f, sort=sort,
                  citation_weight=citation_weight)


def max_query_chars() -> int:
    return int(get_config()["search"]["max_query_chars"])


# ---------------------------------------------------------------------------
# Global notices
# ---------------------------------------------------------------------------
def data_age_days(bundle) -> int | None:
    try:
        ts = pd.Timestamp(bundle.meta.get("generated_at"))
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return int((pd.Timestamp.now(tz="UTC").tz_convert(None) - ts).days)


def global_notices(bundle) -> None:
    """Staleness, degraded search and synthetic-data banners."""
    age = data_age_days(bundle)
    limit = int(get_config()["monitoring"]["stale_after_days"])
    if age is not None and age > limit:
        note(f"<b>This data may be out of date.</b> The current release was built {age} days ago "
             f"(refresh expected every {max(1, limit // 2)} days).")
    if bundle.embedder is None:
        note("<b>Semantic search is temporarily unavailable.</b> Results use keyword matching "
             "(BM25) until the embedding model loads again.")
    snap = bundle.meta.get("snapshot", {})
    if snap.get("source") == "synthetic":
        note("<b>Demo data.</b> This landscape is built from synthetic papers, not real arXiv records.")


def attribution_footer() -> None:
    st.markdown(
        "<div style='margin-top:2.5rem;padding-top:1rem;border-top:1px solid #e2e5ea;"
        "color:#566070;font-size:.8125rem;line-height:1.6'>"
        "Thank you to arXiv for use of its open access interoperability. Paper metadata from "
        "<a href='https://arxiv.org' target='_blank' rel='noopener'>arXiv</a>; abstracts remain "
        "under each paper's own licence, follow the links to read them on arXiv. Citation data from "
        "<a href='https://openalex.org' target='_blank' rel='noopener'>OpenAlex</a> (CC0). "
        "ScholarGrid is an exploration tool, not a judgment of research value."
        "</div>", unsafe_allow_html=True)


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def area_info(clusters: dict, cid) -> dict:
    return clusters.get(cid) or clusters.get(str(cid)) or {}


def area_name(clusters: dict, cid) -> str:
    if cid is None or int(cid) == -1:
        return "Unclustered"
    info = area_info(clusters, int(cid))
    if info.get("keywords"):
        return pretty_label(info["keywords"])
    return info.get("label", f"Area {cid}")


# Kept for callers that still use the old name.
cluster_name = area_name


def area_colors(clusters: dict) -> dict:
    """Stable colour per area (largest areas get the first palette slots)."""
    order = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    return {int(c["cluster_id"]): PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}


def growth_of(growth: dict, cid):
    return growth["clusters"].get(str(cid)) or growth["clusters"].get(cid)


def history_days(bundle) -> int:
    g = bundle.growth
    if "history_days" in g:
        return int(g["history_days"])
    d = bundle.df["date"]
    return int((d.max() - d.min()).days) + 1 if len(d) else 0


def growth_ready(bundle) -> bool:
    """True when the corpus spans enough time for window-based growth."""
    g = bundle.growth
    if "sufficient_history" in g:
        return bool(g["sufficient_history"])
    return history_days(bundle) >= 2 * int(g["default_window"]) * 30.4


def growth_window(growth: dict, cid) -> dict:
    g = growth_of(growth, cid)
    if not g:
        return {}
    return g["windows"].get(str(growth["default_window"]), {}) or {}


def growth_reliable(growth: dict, cid) -> bool:
    """True only when the cluster passes every gate (history, minimum counts,
    confidence interval, stability). Bundles built before v2 never qualify."""
    g = growth_of(growth, cid)
    return bool(g and g.get("reliable"))


def relative_growth(growth: dict, cid):
    return growth_window(growth, cid).get("relative_growth")


def growth_text(growth: dict, cid) -> str:
    """Plain-language growth summary with window, counts and interval."""
    w = growth_window(growth, cid)
    months = growth.get("default_window")
    counts = f"{w.get('recent_count', 0)} papers in the last {months} months vs {w.get('previous_count', 0)} before"
    if not growth_reliable(growth, cid) or w.get("relative_growth") is None:
        return f"{counts}. Trend not reliable yet."
    lo, hi = (w.get("ci") or [None, None])[:2]
    ci = f" ({int(round(growth['gate']['ci_level'] * 100))}% CI {lo:.2f}–{hi:.2f})" if lo is not None else ""
    return f"{w['relative_growth']:.2f}× the CS-wide rate{ci}; {counts}."


def fmt_date(d) -> str:
    try:
        ts = pd.Timestamp(d)
    except Exception:
        return str(d)
    if pd.isna(ts):
        return "—"
    return f"{ts.day} {ts:%b %Y}"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def render_top_nav(pages: dict) -> None:
    with st.container(key="topnav"):
        st.markdown("<div class='sg-brand'>ScholarGrid <span class='sg-brand-sub'>Research, with direction.</span></div>",
                    unsafe_allow_html=True)
        cols = st.columns([1.2, 1.2, 1.2, .8], gap="small")
        for col, key in zip(cols[:3], ["home", "explore", "library"]):
            if key in pages:
                col.page_link(pages[key])
        with cols[3], st.popover("More", use_container_width=True):
            for key in ["areas", "trends", "leads", "about"]:
                if key in pages:
                    st.page_link(pages[key])
            st.caption("Start with Find papers. Use these tools when you want deeper context.")


def goto(key: str) -> None:
    pages = st.session_state.get("_pages", {})
    if key in pages:
        st.switch_page(pages[key])


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, lead: str | None = None) -> None:
    lead_html = f"<p class='sg-lead'>{html.escape(lead)}</p>" if lead else ""
    st.markdown(
        f"<div class='sg-page-head'><div class='sg-eyebrow'>{html.escape(eyebrow)}</div>"
        f"<div class='sg-title'>{html.escape(title)}</div>{lead_html}</div>",
        unsafe_allow_html=True)


def panel(key: str):
    """A white section panel; use as a context manager."""
    return st.container(key=f"panel-{key}")


def panel_header(title: str, meta: str | None = None) -> None:
    meta_html = f"<span class='sg-ph-m'>{html.escape(meta)}</span>" if meta else ""
    st.markdown(f"<div class='sg-ph'><span class='sg-ph-t'>{html.escape(title)}</span>"
                f"{meta_html}</div>", unsafe_allow_html=True)


def note(text_html: str) -> None:
    st.markdown(f"<div class='sg-note'>{text_html}</div>", unsafe_allow_html=True)


def stat_tiles(tiles: Sequence[tuple]) -> None:
    cells = "".join(
        f"<div class='sg-tile'><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='k'>{html.escape(str(k))}</div></div>"
        for v, k in tiles)
    st.markdown(f"<div class='sg-tiles'>{cells}</div>", unsafe_allow_html=True)


def growth_badge(growth: dict, cid, ready: bool) -> str:
    """Reliability-aware badge: a direction only when the interval excludes 1."""
    if not ready:
        return ""
    if not growth_reliable(growth, cid):
        return "<span class='sg-badge sg-flat'>trend not reliable</span>"
    w = growth_window(growth, cid)
    direction, rel = w.get("direction"), w.get("relative_growth")
    if direction == "growing":
        return f"<span class='sg-badge sg-up'>growing · {rel:.1f}×</span>"
    if direction == "declining":
        return f"<span class='sg-badge sg-flat'>slowing · {rel:.1f}×</span>"
    return "<span class='sg-badge sg-flat'>no clear change</span>"


def keyword_pills(words: Iterable[str]) -> str:
    pills = "".join(f"<span class='sg-pill'>{html.escape(str(w))}</span>" for w in words)
    return f"<div class='sg-pills'>{pills}</div>"


@st.cache_resource(show_spinner=False)
def _paper_index(version: str) -> dict:
    df = get_bundle().df
    return {str(a): (str(au or ""), int(c))
            for a, au, c in zip(df["arxiv_id"], df["authors"], df["cited_by_count"])}


def author_line(authors: str, max_names: int = 3) -> str:
    names = [n.strip() for n in str(authors or "").split(";") if n.strip()]
    if not names:
        return ""
    more = f" +{len(names) - max_names}" if len(names) > max_names else ""
    return ", ".join(names[:max_names]) + more


def _highlight(text: str, terms: Sequence[str]) -> str:
    import re

    escaped = html.escape(text)
    if not terms:
        return escaped
    pattern = "|".join(re.escape(html.escape(t)) for t in sorted(terms, key=len, reverse=True))
    return re.sub(rf"(?i)\b({pattern})\b", r"<mark>\1</mark>", escaped)


def paper_card(title: str, arxiv_id: str, *, category: str | None = None,
               date=None, area: str | None = None, score: float | None = None,
               snippet: str | None = None, delay: int = 0, terms: Sequence[str] = ()) -> None:
    index = _paper_index(get_bundle().meta.get("generated_at", "v1"))
    authors, cites = index.get(str(arxiv_id), ("", 0))
    chips = ""
    if cites:
        chips += f"<span class='sg-chip cites'>{cites:,} citation{'s' if cites != 1 else ''}</span>"
    if category:
        chips += f"<span class='sg-chip'>{html.escape(str(category))}</span>"
    if date is not None:
        chips += f"<span class='sg-chip'>{html.escape(fmt_date(date))}</span>"
    if area:
        chips += f"<span class='sg-chip area'>{html.escape(area)}</span>"
    if score is not None:
        chips += f"<span class='sg-score'>Ranking score {score:.3f}</span>"
    body = (
        f"<div class='sg-card' style='animation-delay:{min(delay, 12) * 45}ms'>"
        f"<div class='t'><a href='https://arxiv.org/abs/{html.escape(str(arxiv_id))}' "
        f"target='_blank' rel='noopener'>{html.escape(str(title))}</a></div>"
        + (f"<div class='a'>{html.escape(author_line(authors))}</div>" if authors else "")
        + f"<div class='m'>{chips}</div>"
    )
    if snippet:
        body += f"<div class='s'>{_highlight(snippet, terms)}</div>"
    st.markdown(body + "</div>", unsafe_allow_html=True)


def abstract_snippet(text, n: int = 230) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def skeleton_cards(n: int = 4) -> str:
    one = ("<div class='sg-skel'><i style='width:70%'></i><i style='width:40%'></i>"
           "<i style='width:95%'></i><i style='width:85%'></i></div>")
    return one * n


def skeleton_block() -> str:
    return "<div class='sg-skel-block'></div>"


def html_table(rows: list[dict], columns: list[str], numeric: Sequence[str] = (),
               bar_col: str | None = None) -> None:
    """Readable HTML table (st.dataframe draws on a canvas with tiny text)."""
    peak = max((float(r[bar_col]) for r in rows), default=1.0) if bar_col else 1.0
    head = "".join(f"<th class='{'num' if c in numeric else ''}'>{html.escape(c)}</th>"
                   for c in columns)
    body = []
    for i, r in enumerate(rows):
        cells = []
        for c in columns:
            v = r.get(c, "")
            if c == bar_col and peak > 0:
                w = max(4, int(140 * float(v) / peak))
                cells.append(f"<td class='num'><span class='sg-bar' style='width:{w}px;"
                             f"animation-delay:{min(i, 15) * 30}ms'></span>{html.escape(str(v))}</td>")
            else:
                cls = "num" if c in numeric else ""
                cells.append(f"<td class='{cls}'>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f"<table class='sg-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>", unsafe_allow_html=True)


def apply_chart_style(fig, height: int | None = None):
    fig.update_layout(
        font=dict(family=FONT, size=15, color=INK),
        hoverlabel=dict(font=dict(family=FONT, size=15, color=INK),
                        bgcolor="white", bordercolor=BORDER),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=8, b=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig
