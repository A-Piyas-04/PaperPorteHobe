"""Opportunity quadrants: only reliable growth with a CI above 1 counts as growing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scholargrid.opportunity import opportunity_table, quadrant


def test_quadrant_rules():
    assert quadrant(0.8, True, 1.2) == "Hot & competitive"
    assert quadrant(0.2, True, 1.1) == "Emerging"
    assert quadrant(0.8, False, 1.5) == "Mature"           # unreliable growth is never "growing"
    assert quadrant(0.2, True, 0.9) == "Niche"              # CI includes 1
    assert quadrant(0.5, True, None) == "Mature"


def test_opportunity_table_from_bundle_like():
    class B:
        pass

    b = B()
    dates = pd.date_range("2024-01-01", periods=30, freq="15D")
    b.df = pd.DataFrame({"arxiv_id": [f"p{i}" for i in range(30)], "date": dates})
    b.labels = np.array([0] * 10 + [1] * 20)
    b.clusters_meta = {0: {"label": "Alpha"}, 1: {"label": "Beta"}}
    b.growth = {"default_window": 6, "clusters": {"0": {"reliable": True, "windows": {"6": {"relative_growth": 1.4,
                                                                                           "ci": [1.1, 1.8]}}},
                                                  "1": {"reliable": False, "windows": {"6": {"relative_growth": 2.0}}}}}
    b.gaps = {"areas": {"0": [{"id": "g1"}, {"id": "g2"}]}}
    b.problems = pd.DataFrame({"arxiv_id": ["p1", "p1", "p2"]})
    b.profiles = pd.DataFrame({"arxiv_id": [f"p{i}" for i in range(10)]})
    t = opportunity_table(b).set_index("name")
    assert t.loc["Alpha", "open_problems"] == 2 and t.loc["Alpha", "stated_per_100"] == 30.0
    assert np.isnan(t.loc["Beta", "growth"]) and not t.loc["Beta", "reliable"]
    assert t.loc["Alpha", "crowdedness"] < t.loc["Beta", "crowdedness"]
    assert t.loc["Alpha", "quadrant"] == quadrant(t.loc["Alpha", "crowdedness"], True, 1.1)   # growing
    assert t.loc["Beta", "quadrant"] == "Mature"                                              # crowded, unreliable
