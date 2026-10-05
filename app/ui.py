"""Shared UI layer for the ScholarGrid app.

Holds the global light-editorial stylesheet, small reusable components
(paper cards, area cards, pills, stat tiles), and the data/helper utilities
that every page needs (bundle loading, cluster lookups).

Design direction: clean, light, print-like. Big type, flat bordered cards,
subtle hover lift and a gentle page-entrance fade. No dark/glass surfaces, no
gradient heroes, no glow/spotlight/ring effects.
"""
from __future__ import annotations

import html
import os
import sys
from typing import Iterable

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.artifacts import load_bundle  # noqa: E402
from scholargrid.config import load_config  # noqa: E402

# --- palette ---------------------------------------------------------------
ACCENT = "#2f6feb"
INK = "#1f2430"
MUTED = "#6b7280"
BORDER = "#e6e8ec"
NOISE_COLOR = "#d4d7dd"
PALETTE = ["#2f6feb", "#e8710a", "#1a9e6c", "#d1453b", "#7b57d6", "#c99700",
           "#0b8aa6", "#c2457f", "#6b8e23", "#8a6d3b", "#4b6bdb", "#2aa198"]

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
_CSS = f"""
<style>
  :root {{
    --accent:{ACCENT}; --ink:{INK}; --muted:{MUTED}; --border:{BORDER};
  }}

  /* Base layout + typography ------------------------------------------- */
  html, body, [class*="css"] {{ font-size: 17px; }}
  .block-container {{
    padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1180px;
    animation: sgFade .35s ease both;
  }}
  @keyframes sgFade {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  h1, h2, h3 {{ color: var(--ink); letter-spacing: -0.01em; }}
  h1 {{ font-size: 2.4rem !important; font-weight: 800; line-height: 1.1; }}
  h2 {{ font-size: 1.6rem !important; font-weight: 750; margin-top: .2rem; }}
  h3 {{ font-size: 1.2rem !important; font-weight: 700; }}
  a {{ color: var(--accent); }}
  hr {{ margin: 1.6rem 0; }}

  /* Hide Streamlit chrome we replace with our own nav ------------------ */
  #MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
  [data-testid="stSidebarNav"] {{ display: none; }}

  /* Top navigation bar ------------------------------------------------- */
  .sg-topnav {{
    display: flex; align-items: center; gap: .3rem; flex-wrap: wrap;
    padding: .2rem 0 1rem 0; margin-bottom: .6rem;
    border-bottom: 1px solid var(--border);
  }}
  .sg-brand {{
    font-size: 1.25rem; font-weight: 800; color: var(--ink);
    margin-right: 1.2rem; white-space: nowrap;
  }}
  .sg-topnav [data-testid="stPageLink"] a {{
    padding: .45rem .9rem; border-radius: 10px; font-weight: 600;
    color: var(--muted); transition: background .15s ease, color .15s ease;
  }}
  .sg-topnav [data-testid="stPageLink"] a:hover {{
    background: #eef2fb; color: var(--accent);
  }}
  .sg-topnav [data-testid="stPageLink"] a[aria-current] {{
    background: #eef2fb; color: var(--accent);
  }}

  /* Search input ------------------------------------------------------- */
  [data-testid="stTextInput"] input {{
    height: 3.4rem; font-size: 1.15rem; border-radius: 14px;
    border: 1.5px solid var(--border); padding: 0 1rem;
    transition: border-color .15s ease, box-shadow .15s ease;
  }}
  [data-testid="stTextInput"] input:focus {{
    border-color: var(--accent); box-shadow: 0 0 0 3px #e6eefc;
  }}

  /* Buttons ------------------------------------------------------------ */
  div[data-testid="stButton"] button {{
    border-radius: 12px; font-weight: 650; padding: .55rem 1rem;
    border: 1.5px solid var(--border); transition: all .15s ease;
  }}
  div[data-testid="stButton"] button:hover {{
    border-color: var(--accent); color: var(--accent);
    transform: translateY(-1px);
  }}
  div[data-testid="stButton"] button[kind="primary"] {{
    background: var(--accent); border-color: var(--accent); color: #fff;
  }}
  div[data-testid="stButton"] button[kind="primary"]:hover {{
    filter: brightness(1.05); color: #fff;
  }}

  /* Stat tiles --------------------------------------------------------- */
  .sg-tiles {{ display:flex; gap:1rem; flex-wrap:wrap; }}
  .sg-tile {{
    flex:1 1 150px; background:#fff; border:1px solid var(--border);
    border-radius:16px; padding:1.1rem 1.3rem;
  }}
  .sg-tile .v {{ font-size:2rem; font-weight:800; color:var(--ink); line-height:1; }}
  .sg-tile .k {{ font-size:.85rem; color:var(--muted); margin-top:.35rem;
                 text-transform:uppercase; letter-spacing:.06em; }}

  /* Cards -------------------------------------------------------------- */
  .sg-card {{
    background:#fff; border:1px solid var(--border); border-radius:14px;
    padding:1rem 1.15rem; margin-bottom:.85rem;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  }}
  .sg-card:hover {{
    transform: translateY(-2px); border-color:#d7dbe2;
    box-shadow: 0 6px 20px rgba(31,36,48,.07);
  }}
  .sg-card .t {{ font-size:1.05rem; font-weight:700; line-height:1.35; }}
  .sg-card .t a {{ color:var(--ink); text-decoration:none; }}
  .sg-card .t a:hover {{ color:var(--accent); }}
  .sg-card .m {{ color:var(--muted); font-size:.85rem; margin-top:.35rem; }}
  .sg-card .s {{ color:#4b5563; font-size:.92rem; margin-top:.5rem; line-height:1.45; }}

  /* Area card (clickable grid tile) ------------------------------------ */
  .sg-area .t {{ font-size:1.18rem; }}

  /* Pills + badges ----------------------------------------------------- */
  .sg-pills {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }}
  .sg-pill {{
    background:#f3f5f9; color:#3b4252; border-radius:999px;
    padding:.2rem .7rem; font-size:.82rem; font-weight:600;
  }}
  .sg-badge {{
    display:inline-block; padding:.12rem .6rem; border-radius:999px;
    font-size:.78rem; font-weight:700; margin-left:.35rem; vertical-align:middle;
  }}
  .sg-up {{ background:#e7f5ee; color:#1a7f54; }}
  .sg-flat {{ background:#eef0f3; color:#5b6472; }}

  /* Robustness meter --------------------------------------------------- */
  .sg-meter {{ background:#eef0f3; border-radius:999px; height:.55rem;
               width:100%; overflow:hidden; margin:.3rem 0 .1rem; }}
  .sg-meter > span {{ display:block; height:100%; background:var(--accent);
                      border-radius:999px; }}

  /* Eyebrow + lead text ------------------------------------------------ */
  .sg-eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.08em;
                 text-transform:uppercase; font-size:.8rem; }}
  .sg-lead {{ color:var(--muted); font-size:1.05rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def render_top_nav(pages: dict) -> None:
    """Draw the brand + horizontal page links shown at the top of every page."""
    order = ["home", "map", "areas", "trends", "leads", "about"]
    cols = st.columns([2.2, 1, 1, 1, 1, 1, 1.1], gap="small",
                      vertical_alignment="center")
    cols[0].markdown("<div class='sg-brand'>🔭 ScholarGrid</div>",
                     unsafe_allow_html=True)
    for col, key in zip(cols[1:], order):
        if key in pages:
            col.page_link(pages[key])
    st.markdown("<div style='border-bottom:1px solid var(--border);"
                "margin:.1rem 0 1.3rem;'></div>", unsafe_allow_html=True)


def goto(key: str) -> None:
    """Programmatically switch to another registered page."""
    pages = st.session_state.get("_pages", {})
    if key in pages:
        st.switch_page(pages[key])


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading research landscape…")
def get_bundle():
    cfg = load_config()
    return load_bundle(cfg)


def cluster_name(meta: dict, cid: int) -> str:
    if cid == -1:
        return "Unclustered"
    info = meta.get(cid) or meta.get(str(cid)) or {}
    return info.get("label", f"cluster {cid}")


def growth_of(growth: dict, cid: int):
    return growth["clusters"].get(str(cid)) or growth["clusters"].get(cid)


def relative_growth(growth: dict, cid: int):
    g = growth_of(growth, cid)
    if not g:
        return None
    dw = str(growth["default_window"])
    return g["windows"].get(dw, {}).get("relative_growth")


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def section_title(eyebrow: str, title: str, lead: str | None = None) -> None:
    st.markdown(f"<div class='sg-eyebrow'>{html.escape(eyebrow)}</div>",
                unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if lead:
        st.markdown(f"<p class='sg-lead'>{html.escape(lead)}</p>",
                    unsafe_allow_html=True)


def stat_tiles(tiles: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"<div class='sg-tile'><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='k'>{html.escape(str(k))}</div></div>"
        for v, k in tiles)
    st.markdown(f"<div class='sg-tiles'>{cells}</div>", unsafe_allow_html=True)


def growth_badge(rel) -> str:
    if rel is None:
        return ""
    cls, txt = ("sg-up", "faster than CS") if rel >= 1.0 else ("sg-flat", "slower than CS")
    return f"<span class='sg-badge {cls}'>{rel:g}x · {txt}</span>"


def keyword_pills(words: Iterable[str]) -> str:
    pills = "".join(f"<span class='sg-pill'>{html.escape(str(w))}</span>"
                    for w in words)
    return f"<div class='sg-pills'>{pills}</div>"


def paper_card(title: str, arxiv_id: str, sub: str, snippet: str | None = None) -> None:
    body = (
        f"<div class='sg-card'>"
        f"<div class='t'><a href='https://arxiv.org/abs/{html.escape(str(arxiv_id))}' "
        f"target='_blank' rel='noopener'>{html.escape(str(title))}</a></div>"
        f"<div class='m'>{html.escape(str(sub))}</div>"
    )
    if snippet:
        body += f"<div class='s'>{html.escape(snippet)}</div>"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def abstract_snippet(text, n: int = 220) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"
