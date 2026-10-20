"""Shared UI layer for the ScholarGrid app.

Global light-editorial stylesheet, loading/transition effects, reusable
components (panels, cards, pills, tables, skeletons) and the data helpers every
page needs.

Design rules: a 16px type scale with clear heading steps, white section
panels on a soft grey page so sections are clearly separated, flat bordered
cards with a small hover lift, fade/slide entrances, a thin top progress bar
while the app is working. No dark/glass surfaces, gradient heroes, glows,
spotlights or ring effects.

Session keys shared across the research journey:

| Key           | Set by       | Meaning                                              |
|---------------|--------------|------------------------------------------------------|
| topic_query   | start/topic  | the active topic text                                |
| ws_key        | topic        | cache key of the active workspace                    |
| ws_selected   | topic        | selected sub-topic id                                |
| j_*           | each page    | journey completion flags (j_topic, j_read, j_gaps, j_plan, j_project) |
| topic_history | start        | recent topics this session                           |
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

TOPIC_EXAMPLES = [
    "hallucination in retrieval-augmented generation",
    "graph neural networks for drug discovery",
    "privacy attacks on federated learning",
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

  /* ---- Guided tour -----------------------------------------------------
     Deliberately off-theme: the app is light panels + blue accent, so the tour
     is a dark console with a warm accent. It should never read as page content.
  ---------------------------------------------------------------------- */
  .st-key-tour-dock, .st-key-tour-invite {
    position: relative; overflow: hidden;
    background:
      radial-gradient(circle, rgba(255,255,255,.045) 1px, transparent 1.2px) 0 0/18px 18px,
      linear-gradient(145deg, #161f33 0%, #1d2740 60%, #202b46 100%);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 16px;
    padding: 1.5rem 1.75rem 1.3rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 44px rgba(10,16,30,.38), inset 0 1px 0 rgba(255,255,255,.06);
    animation: sgRise .45s cubic-bezier(.2,.7,.2,1) both;
  }
  /* Warm rail across the top: the tour's signature. */
  .st-key-tour-dock::before, .st-key-tour-invite::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
    background: linear-gradient(90deg, #ffb020, #ff7d4d 55%, transparent);
  }
  .sg-tour-head { display: flex; align-items: center; gap: .7rem;
                  flex-wrap: wrap; margin-bottom: .2rem; }
  .sg-tour-tag { background: #ffb020; color: #241803; font-size: .6875rem;
                 font-weight: 900; letter-spacing: .12em; text-transform: uppercase;
                 padding: .25rem .65rem; border-radius: 999px; }
  .sg-tour-count { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
                   color: #8fa3c4; font-size: .78rem; font-weight: 700;
                   letter-spacing: .12em; text-transform: uppercase; }
  .sg-tour-title { color: #f4f7fd; font-size: 1.4rem; font-weight: 800;
                   line-height: 1.3; letter-spacing: -.01em; margin: .55rem 0 .45rem; }
  .sg-tour-body { color: #a9b9d4; font-size: 1rem; line-height: 1.7; margin: 0 0 .9rem; }
  .st-key-tour-dock b, .st-key-tour-invite b { color: #e9f0fb; }
  .sg-tour-do { background: rgba(255,176,32,.10); border-left: 3px solid #ffb020;
                border-radius: 0 10px 10px 0; padding: .65rem .95rem;
                color: #ffd79a; font-size: .9375rem; font-weight: 650; margin-bottom: 1rem; }
  /* The tour's own step rail (Streamlit's progress bar would look like the app). */
  .sg-tour-dots { display: flex; align-items: center; gap: 6px; margin: 0 0 1.1rem; }
  .sg-tour-dots i { display: block; width: 22px; height: 4px; border-radius: 2px;
                    background: rgba(255,255,255,.14); }
  .sg-tour-dots i.done { background: rgba(255,176,32,.55); }
  .sg-tour-dots i.now { background: #ffb020; box-shadow: 0 0 0 3px rgba(255,176,32,.18); }
  /* Buttons inside the tour follow the tour, not the site. */
  .st-key-tour-dock [data-testid="stBaseButton-secondary"],
  .st-key-tour-invite [data-testid="stBaseButton-secondary"] {
    background: transparent !important; border: 1px solid rgba(255,255,255,.20) !important;
  }
  .st-key-tour-dock [data-testid="stBaseButton-secondary"] p,
  .st-key-tour-invite [data-testid="stBaseButton-secondary"] p {
    color: #cfdcf0 !important; font-weight: 650 !important;
  }
  .st-key-tour-dock [data-testid="stBaseButton-secondary"]:hover,
  .st-key-tour-invite [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,176,32,.12) !important; border-color: #ffb020 !important;
    transform: none !important;
  }
  .st-key-tour-dock [data-testid="stBaseButton-secondary"]:hover p,
  .st-key-tour-invite [data-testid="stBaseButton-secondary"]:hover p { color: #ffd79a !important; }
  .st-key-tour-dock [data-testid="stBaseButton-secondary"]:disabled p { color: #5d6e8c !important; }
  .st-key-tour-dock [data-testid="stBaseButton-primary"],
  .st-key-tour-invite [data-testid="stBaseButton-primary"],
  .st-key-tour-dock [data-testid="stButton"] button[kind="primary"],
  .st-key-tour-invite [data-testid="stButton"] button[kind="primary"] {
    background: #ffb020 !important; border: 1px solid #ffb020 !important;
  }
  .st-key-tour-dock [data-testid="stBaseButton-primary"] p,
  .st-key-tour-invite [data-testid="stBaseButton-primary"] p,
  .st-key-tour-dock [data-testid="stButton"] button[kind="primary"] p,
  .st-key-tour-invite [data-testid="stButton"] button[kind="primary"] p {
    color: #241803 !important; font-weight: 800 !important;
  }
  .st-key-tour-dock [data-testid="stBaseButton-primary"]:hover,
  .st-key-tour-invite [data-testid="stBaseButton-primary"]:hover,
  .st-key-tour-dock [data-testid="stButton"] button[kind="primary"]:hover,
  .st-key-tour-invite [data-testid="stButton"] button[kind="primary"]:hover {
    background: #ffc247 !important; border-color: #ffc247 !important;
  }

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
  .sg-chip.new { background: #e5f4ec; color: #17734c; }
  .sg-chip.src { background: transparent; color: var(--muted); font-weight: 600; padding-left: 0; }
  .sg-section { font-size: 1.125rem; font-weight: 800; color: var(--ink); margin: 1.4rem 0 .2rem; }
  .sg-section span { color: var(--muted); font-weight: 600; font-size: .9375rem; margin-left: .5rem; }
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
  .sg-journey.five { grid-template-columns: repeat(5, 1fr); gap: 18px; }
  .sg-journey-bar { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem; margin: 0 0 1rem;
                    font-size: .875rem; color: var(--muted); }
  .sg-journey-bar i { font-style: normal; color: #b6bcc7; }
  .sg-jstep { padding: .15rem .55rem; border-radius: 999px; border: 1px solid transparent; }
  .sg-jstep.done { color: #1a7f55; }
  .sg-jstep.now { color: var(--ink); font-weight: 700; border-color: #c4d5ee; background: #f3f6fd; }
  .sg-subtopic-n { font-size: .8125rem; color: var(--muted); margin: .25rem 0 .35rem; }
  .sg-concepts { display: grid; grid-template-columns: minmax(8rem, 1fr) 3fr; gap: .35rem 1rem; margin: .6rem 0; }
  .sg-concepts dt { font-weight: 700; }
  .sg-concepts dd { margin: 0; color: #333b49; }
  .sg-gap-statement { font-size: 1.0625rem; line-height: 1.55; margin: .3rem 0 .5rem; color: var(--ink); }
  .sg-caveat { background: #fff8e8; border: 1px solid #f0dfb5; border-radius: 12px; padding: .6rem .8rem;
               margin: .4rem 0; font-size: .9375rem; }
  .sg-caveat ul { margin: .3rem 0 0 1.1rem; padding: 0; }
  .sg-score-bars { display: flex; flex-wrap: wrap; gap: .4rem .8rem; align-items: center; font-size: .8125rem;
                   color: var(--muted); margin: .3rem 0 .4rem; }
  .sg-score-part { position: relative; isolation: isolate; padding: .1rem .45rem; border: 1px solid var(--border);
                   border-radius: 6px; overflow: hidden; }
  .sg-score-part i { position: absolute; left: 0; top: 0; bottom: 0; background: #e3ecfd; z-index: -1; }
  .sg-evidence { padding: .45rem 0; border-bottom: 1px solid var(--border); font-size: .9375rem; }
  .sg-evidence blockquote { margin: .35rem 0; padding-left: .7rem; border-left: 3px solid #c4d5ee; color: #333b49; }
  @media (max-width: 700px) {
    .sg-journey, .sg-journey.five { grid-template-columns: 1fr; gap: 12px; }
    .sg-concepts { grid-template-columns: 1fr; }
    .sg-brand-sub { display: none; }
    .st-key-tour-dock, .st-key-tour-invite { padding: 1.15rem 1.2rem 1rem; }
    .sg-tour-title { font-size: 1.15rem; }
    .sg-tour-dots i { width: 14px; }
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


@st.cache_data(show_spinner=False, ttl=20)
def _release_key() -> str:
    """Changes when a new release is published (e.g. by the background grow job)."""
    from scholargrid.release import resolve_bundle_dir

    try:
        path = resolve_bundle_dir(get_config())
        meta = os.path.join(path, "meta.json")
        return f"{path}|{os.path.getmtime(meta) if os.path.exists(meta) else 0}"
    except Exception:
        return "unknown"


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_bundle(release_key: str):
    bundle = load_bundle(get_config())
    if get_config()["app"]["pre_warm"] and bundle.embedder is not None:
        with contextlib.suppress(Exception):
            bundle.embedder.encode_query("warm up")
    return bundle


def get_bundle():
    return _load_bundle(_release_key())


def _clear_bundle() -> None:
    _release_key.clear()
    _load_bundle.clear()


get_bundle.clear = _clear_bundle


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


@st.cache_resource(show_spinner=False, max_entries=1)
def get_searcher(version: str):
    """Collection + live-store searcher (one per bundle version, shared by sessions)."""
    from scholargrid.live_search import LiveSearcher

    return LiveSearcher(get_config(), get_bundle())


@st.cache_data(show_spinner=False, max_entries=512)
def run_search(version: str, query: str, top_k: int, filters: tuple = (), sort: str = "relevance",
               citation_weight: float = 0.0) -> dict:
    """Cached collection-only search (used by the research map)."""
    from scholargrid.search import SearchFilters, search

    b = get_bundle()
    f = SearchFilters(*filters) if filters else None
    return search(query, b.embedder, b.embeddings, b.df, b.labels, b.clusters_meta, top_k=top_k,
                  bm25=b.bm25, index=b.index, cfg=get_config(), filters=f, sort=sort,
                  citation_weight=citation_weight, phrase_index=b.phrase, acronyms=b.acronyms)


def max_query_chars() -> int:
    return int(get_config()["search"]["max_query_chars"])


@st.cache_resource(show_spinner=False)
def _llm_client(config_hash: str):
    from scholargrid.llm import get_llm as _get

    return _get(get_config())


def get_llm():
    """The LLM client every page passes into library calls (never call scholargrid.llm.get_llm in views)."""
    return _llm_client(get_config().config_hash())


@st.cache_resource(show_spinner=False)
def get_work_lookup():
    """One OpenAlex work lookup (with its cache) shared by every session."""
    from scholargrid.foundations import WorkLookup

    cfg = get_config()
    return WorkLookup(cfg, synthetic=get_bundle().meta.get("snapshot", {}).get("source") == "synthetic")


# ---------------------------------------------------------------------------
# Durable projects (plan 08); session-only fallback when persistence is off
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _repo(config_hash: str):
    from scholargrid.persistence import get_repo as _get

    return _get(get_config())


def get_repo():
    """The project repository (migrated once per process), or ``None`` when persistence is disabled."""
    if not get_config()["persistence"]["enabled"]:
        return None
    try:
        return _repo(get_config().config_hash())
    except Exception as exc:
        from scholargrid.monitoring import capture_exception

        capture_exception(exc)
        st.session_state["_repo_error"] = f"{type(exc).__name__}"
        return None


def _bump() -> None:
    st.session_state["_proj_rev"] = st.session_state.get("_proj_rev", 0) + 1


def project_access():
    from project_access import active_access

    repo = get_repo()
    return None if repo is None else active_access(get_config(), repo)


@st.cache_data(show_spinner=False, ttl=600, max_entries=256)
def _project_items(pid: str, token: str, user_id: str, rev: int) -> dict:
    from scholargrid.persistence.repo import Access, AccessDenied

    repo = get_repo()
    acc = Access(user_id=user_id or None, project_id=pid, token=token or None)
    try:
        return {"papers": repo.papers(pid, acc), "gaps": repo.gaps(pid, acc), "ok": True}
    except AccessDenied:
        return {"papers": [], "gaps": [], "ok": False}


def project_items() -> dict:
    """Saved papers and gaps of the active project (cached per project revision)."""
    acc = project_access()
    if acc is None or not acc.project_id:
        return {"papers": [], "gaps": [], "ok": acc is not None}
    return _project_items(acc.project_id, acc.token or "", acc.user_id or "", int(st.session_state.get("_proj_rev", 0)))


def saved_paper_ids() -> set:
    if get_repo() is None:
        return set((st.session_state.get("saved_papers") or {}).keys())
    return {p["paper_id"] for p in project_items()["papers"]}


def saved_gap_ids() -> set:
    if get_repo() is None:
        return set((st.session_state.get("saved_gaps") or {}).keys())
    return {g["gap_id"] for g in project_items()["gaps"]}


def saved_gaps() -> list[dict]:
    if get_repo() is None:
        return list((st.session_state.get("saved_gaps") or {}).values())
    return [g["payload"] for g in project_items()["gaps"]]


def paper_save_label(saved: bool) -> str:
    where = "project" if get_repo() is not None else "reading list"
    return f"Remove from {where}" if saved else f"Save to {where}"


_RECORD_FIELDS = ["paper_id", "arxiv_id", "doi", "url", "title", "authors", "venue", "date",
                  "primary_category", "cited_by_count", "abstract", "sources"]


def paper_record(paper_id: str, record: dict | None = None) -> dict | None:
    """A saveable dict for a collection or live paper."""
    if record:
        row = dict(record)
    else:
        df = get_bundle().df
        rows = df.loc[df.arxiv_id.astype(str) == paper_id]
        if not rows.empty:
            row = rows.iloc[0].to_dict()
            row.update(url=paper_url(paper_id), sources=["arxiv"])
        else:
            row = get_searcher(bundle_version()).paper(paper_id) or {}
    if not row:
        return None
    row = {k: row.get(k, "") for k in _RECORD_FIELDS}
    row["paper_id"] = paper_id
    row["date"] = str(row["date"] or "")[:10]
    return row


def _project_id_for_write():
    from project_access import active_access, ensure_project

    cfg, repo = get_config(), get_repo()
    acc = active_access(cfg, repo)
    pid = acc.project_id or ensure_project(cfg, repo, title=str(st.session_state.get("topic_query") or ""),
                                           topic=str(st.session_state.get("topic_query") or ""),
                                           bundle_version=bundle_version())
    return repo, pid, active_access(cfg, repo)


def toggle_paper(paper_id: str, record: dict | None = None, source: str = "search") -> None:
    """Add or remove a paper (collection or found online) from the project / reading list."""
    if get_repo() is None:
        saved = dict(st.session_state.get("saved_papers", {}))
        if paper_id in saved:
            del saved[paper_id]
        else:
            row = paper_record(paper_id, record)
            if row:
                saved[paper_id] = row
        st.session_state.saved_papers = saved
        return
    try:
        repo, pid, acc = _project_id_for_write()
        if paper_id in saved_paper_ids():
            repo.remove_paper(pid, acc, paper_id)
        else:
            row = paper_record(paper_id, record)
            if row:
                repo.add_paper(pid, acc, row, source=source)
    except ValueError as exc:
        st.session_state["_save_error"] = str(exc)
    _bump()


def toggle_gap(card: dict) -> None:
    if get_repo() is None:
        saved = dict(st.session_state.get("saved_gaps", {}))
        if card["id"] in saved:
            del saved[card["id"]]
        else:
            saved[card["id"]] = card
        st.session_state.saved_gaps = saved
        return
    try:
        repo, pid, acc = _project_id_for_write()
        if card["id"] in saved_gap_ids():
            repo.remove_gap(pid, acc, card["id"])
        else:
            repo.add_gap(pid, acc, card)
    except ValueError as exc:
        st.session_state["_save_error"] = str(exc)
    _bump()


def save_plan(doc: dict) -> bool:
    if get_repo() is None:
        saved = dict(st.session_state.get("saved_plans", {}))
        saved[doc["id"]] = doc
        st.session_state.saved_plans = saved
        return False
    repo, pid, acc = _project_id_for_write()
    repo.add_plan(pid, acc, doc)
    _bump()
    return True


def project_notices() -> None:
    """The 'bookmark this link' notice after an anonymous project is created, and save errors."""
    from project_access import project_link

    err = st.session_state.pop("_save_error", None)
    if err:
        note(f"<b>Couldn't save.</b> {html.escape(err)}")
    if st.session_state.get("show_link_notice") and st.session_state.get("proj_id"):
        with st.container(key="link-notice"):
            note("<b>Your project is saved. Bookmark this page.</b> Anyone with this link can view and edit the "
                 "project.")
            st.code(project_link(), language=None)
            if st.button("Got it", key="link_notice_ok"):
                st.session_state["show_link_notice"] = False
                st.rerun()


def get_workspace(query: str):
    """The topic workspace for ``query``: session cache, then disk cache, else build it with progress."""
    from scholargrid.workspace import build_workspace, load_cached, workspace_key

    cfg = get_config()
    v = bundle_version()
    key = workspace_key(query, v, cfg)
    cache = st.session_state.setdefault("_ws", {})
    if key in cache:
        st.session_state["ws_key"] = key
        return cache[key]
    ws = load_cached(key, cfg)
    if ws is None:
        with st.status("Mapping your topic…", expanded=True) as status:
            bar = st.progress(0.0)

            def progress(msg: str, frac: float) -> None:
                status.update(label=msg)
                bar.progress(min(1.0, max(0.0, float(frac))))

            ws = build_workspace(query, get_bundle(), get_searcher(v), cfg, bundle_version=v, progress=progress,
                                 lookup=get_work_lookup(), llm=get_llm())
            status.update(label="Done", state="complete", expanded=False)
    if len(cache) >= 8:
        cache.pop(next(iter(cache)))
    cache[key] = ws
    st.session_state["ws_key"] = key
    return ws


def require_workspace():
    """The active workspace, or ``None`` after showing a 'choose a topic first' note."""
    q = str(st.session_state.get("topic_query", "") or "").strip()
    if not q:
        note("<b>No topic yet.</b> Start by describing what you want to research.")
        if st.button("Choose a topic", type="primary", key="need_topic"):
            goto("start")
        return None
    try:
        return get_workspace(q)
    except Exception as exc:
        from scholargrid.monitoring import capture_exception

        capture_exception(exc)
        note("<b>We couldn't map this topic right now.</b> The error has been logged. Try again in a moment, "
             "or try a different phrasing.")
        if st.button("Try again", key="ws_retry"):
            st.rerun()
        return None


def topic_examples() -> list[str]:
    cats = get_config()["scope"]["categories"]
    if cats != "all" and any(str(c).startswith("cs.CL") for c in cats):
        return ["hallucination in retrieval-augmented generation", "low-resource machine translation",
                "evaluating large language model reasoning"]
    return TOPIC_EXAMPLES


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
        "<a href='https://arxiv.org' target='_blank' rel='noopener'>arXiv</a>, "
        "<a href='https://openalex.org' target='_blank' rel='noopener'>OpenAlex</a> (CC0) and "
        "<a href='https://www.semanticscholar.org' target='_blank' rel='noopener'>Semantic Scholar</a>; "
        "abstracts remain under each paper's own licence, follow the links to read them at the source. "
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


CITATION_COVERAGE_MIN = 0.1


def citation_coverage(bundle) -> float:
    """Share of collected papers with at least one citation."""
    counts = bundle.meta.get("counts") or {}
    papers = int(counts.get("papers") or len(bundle.df))
    if not papers:
        return 0.0
    cited = counts.get("with_citations")
    if cited is None:
        cited = int((bundle.df["cited_by_count"] > 0).sum())
    return float(cited) / papers


def citations_ready(bundle) -> bool:
    """True when enough papers have citation data for citation sorts/filters to mean anything."""
    return citation_coverage(bundle) >= CITATION_COVERAGE_MIN


def velocity_ready(bundle) -> bool:
    """True when enough cited papers show citation change between snapshots."""
    if not citations_ready(bundle) or "citation_velocity" not in bundle.df.columns:
        return False
    cited = bundle.df["cited_by_count"] > 0
    return float((bundle.df.loc[cited, "citation_velocity"] > 0).mean()) >= CITATION_COVERAGE_MIN


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
JOURNEY = ["start", "topic", "reading", "gaps", "plan", "project"]
MORE = ["home", "explore", "areas", "trends", "library", "leads", "about"]
_JOURNEY_STEPS = [("topic", "Topic", "j_topic"), ("reading", "Read first", "j_read"),
                  ("gaps", "Open problems", "j_gaps"), ("plan", "Plan", "j_plan"),
                  ("project", "Project", "j_project")]


def render_top_nav(pages: dict) -> None:
    from views import tour

    with st.container(key="topnav"):
        st.markdown("<div class='sg-brand'>ScholarGrid <span class='sg-brand-sub'>From a topic to a "
                    "research question.</span></div>", unsafe_allow_html=True)
        steps = [k for k in JOURNEY if k in pages]
        cols = st.columns([1] * len(steps) + [0.8, 0.8], gap="small")
        for col, key in zip(cols, steps):
            col.page_link(pages[key])
        durable = get_repo() is not None
        with cols[len(steps)], st.popover("More", use_container_width=True):
            for key in MORE:
                if key in pages and not (key == "library" and durable):
                    st.page_link(pages[key])
            st.caption("Follow the steps from Start. These tools give deeper context.")
            _account_controls()
        with cols[len(steps) + 1]:
            tour.header_control()


def _account_controls() -> None:
    """Sign in / out (only with persistence.auth: oidc and Streamlit auth configured)."""
    cfg = get_config()
    if cfg["persistence"]["auth"] != "oidc" or get_repo() is None:
        return
    user = getattr(st, "user", None)
    try:
        logged_in = bool(user is not None and user.is_logged_in)
    except Exception:
        return
    if logged_in:
        st.caption(f"Signed in as {html.escape(str(getattr(user, 'name', '') or getattr(user, 'email', '')))}")
        if st.button("Sign out", key="auth_logout"):
            st.logout()
    elif st.button("Sign in", key="auth_login"):
        provider = cfg["persistence"]["oidc_provider"]
        if provider:
            st.login(provider)
        else:
            st.login()


def journey_bar(current: str) -> None:
    """Thin breadcrumb: 1 Topic · 2 Read first · 3 Open problems · 4 Plan · 5 Project."""
    pages = st.session_state.get("_pages") or {}
    parts = []
    for n, (key, label, flag) in enumerate(_JOURNEY_STEPS, start=1):
        if pages and key not in pages:
            continue
        done = bool(st.session_state.get(flag))
        cls = "now" if key == current else ("done" if done else "")
        mark = "✓ " if done and key != current else f"{n} "
        aria = " aria-current='step'" if key == current else ""
        parts.append(f"<span class='sg-jstep {cls}'{aria}>{mark}{html.escape(label)}</span>")
    st.markdown("<nav class='sg-journey-bar' aria-label='Research journey'>" + "<i>·</i>".join(parts)
                + "</nav>", unsafe_allow_html=True)


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


def paper_url(paper_id: str, url: str | None = None) -> str:
    """Link for a paper: its own URL, else arXiv for bare arXiv IDs, else doi.org."""
    pid = str(paper_id or "")
    if url:
        return str(url)
    if pid.startswith("doi:"):
        return f"https://doi.org/{pid[4:]}"
    if ":" in pid:
        return ""
    return f"https://arxiv.org/abs/{pid}"


_SOURCE_NAMES = {"arxiv": "arXiv", "openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar"}
NEW_CHIP = "New · found online"
_NEW_TIP = ("Found by a live lookup, not in the analyzed collection. Not part of the areas, "
            "map or trends until the next data build.")
OUTSIDE_CHIP = "Outside the collection · via OpenAlex"
_OUTSIDE_TIP = ("Cited by many papers here but not itself in the collection (often older than its "
                "start date), so the map and areas don't show it.")


def paper_card(title: str, paper_id: str, *, url: str | None = None, authors: str | None = None,
               cites: int | None = None, category: str | None = None, venue: str | None = None,
               date=None, area: str | None = None, score: float | None = None,
               snippet: str | None = None, delay: int = 0, terms: Sequence[str] = (),
               sources: Sequence[str] = (), new: bool = False, outside: bool = False) -> None:
    if authors is None or cites is None:
        index = _paper_index(get_bundle().meta.get("generated_at", "v1"))
        a, c = index.get(str(paper_id), ("", 0))
        authors = a if authors is None else authors
        cites = c if cites is None else cites
    chips = ""
    if new:
        chips += f"<span class='sg-chip new' title='{html.escape(_NEW_TIP)}'>{NEW_CHIP}</span>"
    if outside:
        chips += f"<span class='sg-chip src' title='{html.escape(_OUTSIDE_TIP)}'>{OUTSIDE_CHIP}</span>"
    if cites:
        chips += f"<span class='sg-chip cites'>{cites:,} citation{'s' if cites != 1 else ''}</span>"
    if category:
        chips += f"<span class='sg-chip'>{html.escape(str(category))}</span>"
    if venue and not category:
        chips += f"<span class='sg-chip'>{html.escape(str(venue)[:60])}</span>"
    if date is not None and str(date):
        chips += f"<span class='sg-chip'>{html.escape(fmt_date(date) if len(str(date)) > 4 else str(date))}</span>"
    if area:
        chips += f"<span class='sg-chip area'>{html.escape(area)}</span>"
    shown = [_SOURCE_NAMES.get(s, s) for s in sources if s in _SOURCE_NAMES]
    if shown:
        chips += f"<span class='sg-chip src'>via {html.escape(' · '.join(shown))}</span>"
    if score is not None:
        chips += f"<span class='sg-score'>Ranking score {score:.3f}</span>"
    link = paper_url(paper_id, url)
    title_html = (f"<a href='{html.escape(link)}' target='_blank' rel='noopener'>{html.escape(str(title))}</a>"
                  if link else html.escape(str(title)))
    body = (
        f"<div class='sg-card' style='animation-delay:{min(delay, 12) * 45}ms'>"
        f"<div class='t'>{title_html}</div>"
        + (f"<div class='a'>{html.escape(author_line(authors))}</div>" if authors else "")
        + f"<div class='m'>{chips}</div>"
    )
    if snippet:
        body += f"<div class='s'>{_highlight(snippet, terms)}</div>"
    st.markdown(body + "</div>", unsafe_allow_html=True)


READING_STEPS = {"survey": "Start with an overview", "foundation": "Foundations",
                 "recent": "Key recent work", "frontier": "The frontier"}


def reading_path_list(path: list[dict], key: str = "rp") -> None:
    """Render a reading path (surveys -> foundations -> key recent -> frontier) as grouped paper cards."""
    n = 0
    for step, label in READING_STEPS.items():
        items = [p for p in path if p.get("step") == step]
        if not items:
            continue
        st.markdown(f"<div class='sg-section'>{html.escape(label)}</div>", unsafe_allow_html=True)
        for p in items:
            snippet = p.get("why_read") or p.get("reason") or ""
            if p.get("what_you_learn"):
                snippet += f" You'll learn: {p['what_you_learn']}"
            paper_card(p.get("title", ""), str(p.get("paper_id", "")), url=p.get("url") or None, authors="",
                       cites=int(p.get("cited_by_count") or 0), date=str(p.get("year") or ""),
                       snippet=snippet, delay=n, outside=not p.get("in_corpus", True))
            n += 1


_SCORE_PARTS = [("evidence", "Evidence"), ("novelty", "Novelty"), ("feasibility", "Feasibility"),
                ("specificity", "Specificity"), ("recency", "Recency")]
_LIVE_SOURCES = "arXiv, OpenAlex, Semantic Scholar"


def _ago(ts) -> str:
    import time as _t

    if not ts:
        return "not yet"
    days = int((_t.time() - float(ts)) // 86400)
    return "today" if days <= 0 else ("yesterday" if days == 1 else f"{days} days ago")


def novelty_line(nov: dict) -> tuple[str, bool]:
    """(html, offer a live re-check)."""
    status = nov.get("status", "not_checked")
    scope = nov.get("scope", "none")
    when = _ago(nov.get("checked_at"))
    where = f"collection + {_LIVE_SOURCES}" if scope == "collection+live" else "the collection only"
    related = [h for h in nov.get("hits", []) if h.get("verdict") in ("addresses", "partially")]
    links = ", ".join(f"<a href='{html.escape(h.get('url') or '')}' target='_blank' rel='noopener'>"
                      f"{html.escape(h.get('title', '')[:70])}</a>" for h in related[:3])
    if status == "not_checked":
        return "Novelty not checked yet.", True
    if status == "novel":
        text = f"No paper found that already does this (checked {when}: {where})."
    elif status == "partially_explored":
        text = f"{len(related)} related paper{'s' if len(related) != 1 else ''} found ({links}); checked {when}: {where}."
    else:
        text = f"Already explored by: {links} (checked {when}: {where})."
    if nov.get("method") == "heuristic":
        text += " Judged by similarity only (no AI check)."
    return text, scope != "collection+live"


def gap_card(card: dict, key_prefix: str, *, on_recheck=None, on_save=None, on_plan=None,
             on_feedback=None, saved: bool = False, label: str | None = None) -> None:
    """Render one Gap Card: type, statement, caveats, novelty, score parts, evidence and first steps."""
    from scholargrid.gaps.model import ROLE_LABELS, TYPE_HELP, TYPE_LABELS

    cid = str(card.get("id"))
    typ = str(card.get("type", ""))
    chips = (f"<span class='sg-chip area' title='{html.escape(TYPE_HELP.get(typ, ''))}'>"
             f"{html.escape(TYPE_LABELS.get(typ, typ))}</span>"
             f"<span class='sg-chip'>Confidence: {html.escape(str(card.get('confidence', 'low')))}</span>"
             f"<span class='sg-chip'>Feasibility: {html.escape(str(card.get('feasibility', 'medium')))}</span>")
    if card.get("beginner_friendly"):
        chips += "<span class='sg-chip cites'>Beginner-friendly</span>"
    if label:
        chips += f"<span class='sg-chip src'>{html.escape(label)}</span>"
    if not (card.get("generated_by") or {}).get("synthesized", True):
        chips += "<span class='sg-chip'>Template wording</span>"
    with st.container(key=f"gap-{key_prefix}-{cid}"):
        st.markdown(f"<div class='sg-area-t'>{html.escape(str(card.get('title', '')))}</div>"
                    f"<div class='m' style='margin:.35rem 0'>{chips}</div>"
                    f"<p class='sg-gap-statement'>{html.escape(str(card.get('statement', '')))}</p>",
                    unsafe_allow_html=True)
        if card.get("why_it_matters"):
            st.markdown(f"**Why it matters.** {card['why_it_matters']}")
        reasons = [r for r in card.get("why_it_might_not_be_a_gap", []) if r]
        if reasons:
            more = "".join(f"<li>{html.escape(r)}</li>" for r in reasons[1:])
            st.markdown("<div class='sg-caveat'><b>Why this might not be a gap:</b> "
                        f"{html.escape(reasons[0])}{'<ul>' + more + '</ul>' if more else ''}</div>",
                        unsafe_allow_html=True)
        nov_text, offer = novelty_line(card.get("novelty") or {})
        st.markdown(f"<div class='sg-subtopic-n'>{nov_text}</div>", unsafe_allow_html=True)
        scores = card.get("scores") or {}
        explain = scores.get("explain") or {}
        bars = "".join(
            f"<span class='sg-score-part' title='{html.escape(name)}: {html.escape(str(explain.get(k, '')))}'>"
            f"<i style='width:{int(100 * float(scores.get(k, 0) or 0))}%'></i>{html.escape(name)}</span>"
            for k, name in _SCORE_PARTS)
        st.markdown(f"<div class='sg-score-bars' aria-label='Score {float(scores.get('total', 0) or 0):.2f}'>"
                    f"<b>Score {float(scores.get('total', 0) or 0):.2f}</b>{bars}</div>", unsafe_allow_html=True)
        ev = card.get("evidence") or []
        with st.expander(f"Evidence ({len({e.get('paper_id') for e in ev})} papers)"):
            for e in ev:
                year = f" ({e['year']})" if e.get("year") else ""
                role = ROLE_LABELS.get(str(e.get("role")), str(e.get("role")))
                body = (f"<b>{html.escape(str(e.get('eid', '')))}</b> · <a href='{html.escape(str(e.get('url', '')))}' "
                        f"target='_blank' rel='noopener'>{html.escape(str(e.get('title', '')))}</a>{year} · "
                        f"<i>{html.escape(role)}</i>")
                if e.get("quote"):
                    sect = f" — {html.escape(str(e.get('section')))}" if e.get("section") else ""
                    body += f"<blockquote>“{html.escape(str(e['quote']))}”{sect}</blockquote>"
                    if e.get("anchor_url"):
                        body += (f"<a href='{html.escape(str(e['anchor_url']))}' target='_blank' rel='noopener'>"
                                 "Read in context</a>")
                elif e.get("note"):
                    body += f"<div class='sg-subtopic-n'>{html.escape(str(e['note']))}</div>"
                st.markdown(f"<div class='sg-evidence'>{body}</div>", unsafe_allow_html=True)
        with st.expander("How to start"):
            for step in card.get("first_steps") or []:
                st.markdown(f"- {step}")
            if card.get("skills"):
                st.markdown("**Skills:** " + ", ".join(card["skills"]))
            if card.get("data_needs"):
                st.markdown(f"**Data:** {card['data_needs']}")
            st.markdown(f"**Compute:** {card.get('compute', 'unknown')}")
        cols = st.columns(4)
        if on_save is not None:
            cols[0].button("Remove from project" if saved else "Save to project", key=f"{key_prefix}_save_{cid}",
                           on_click=on_save, args=(card,), use_container_width=True)
        if on_plan is not None:
            cols[1].button("Make a starter plan", key=f"{key_prefix}_plan_{cid}", on_click=on_plan, args=(card,),
                           use_container_width=True)
        if on_recheck is not None and offer:
            cols[2].button("Check novelty now", key=f"{key_prefix}_nov_{cid}", on_click=on_recheck, args=(card,),
                           use_container_width=True)
        if on_feedback is not None:
            cols[3].button("Not useful", key=f"{key_prefix}_bad_{cid}", on_click=on_feedback, args=(card,),
                           use_container_width=True)


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
