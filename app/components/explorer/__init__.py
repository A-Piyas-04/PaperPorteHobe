"""Paper-graph explorer: a custom Streamlit component (React + sigma.js).

Release mode serves the prebuilt bundle in ``frontend/dist``. Set
``SCHOLARGRID_DEV=1`` and run ``npm run dev`` in ``frontend/`` to load the
Vite dev server instead (hot reload while editing the UI).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_HERE = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(_HERE, "frontend", "dist")
DEV_MODE = os.environ.get("SCHOLARGRID_DEV") == "1"

if DEV_MODE:
    _component = components.declare_component("scholargrid_explorer", url="http://localhost:5173")
else:
    _component = components.declare_component("scholargrid_explorer", path=DIST_DIR)


def is_built() -> bool:
    return DEV_MODE or os.path.exists(os.path.join(DIST_DIR, "index.html"))


def explorer(data: Dict[str, Any], *, highlight: Optional[Dict] = None,
             focus_area: Optional[int] = None, select: Optional[str] = None,
             height: int = 780, key: str = "explorer") -> Optional[Dict]:
    """Render the explorer. Returns the latest event dict sent by the UI
    (``{"type": "search", "q": ..., "nonce": ...}`` etc.) or ``None``."""
    return _component(data=data, highlight=highlight, focus_area=focus_area,
                      select=select, height=height, key=key, default=None)
