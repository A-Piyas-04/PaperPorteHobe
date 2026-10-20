"""Deterministic synthetic full text (CI / demo). Not real science - planted structure only.

Mirrors the synthetic corpus (``data_ingest._synthetic``) and plants the patterns plans 03 and 06 must
find: recurring stated problems (a small pool, so the same problem appears in >= 3 papers), a missing
(method, task) combination, and one method that is always evaluated on a single dataset.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .model import Section

DATASETS = {"cs.CL": ["SynthGLUE", "SynthQA", "SynthMT"], "cs.CV": ["SynthNet", "SynthCOCO", "SynthSeg"],
            "cs.LG": ["SynthTab", "SynthBench"], "cs.CR": ["SynthMalware", "SynthPriv"],
            "cs.RO": ["SynthSim", "SynthGrasp"], "cs.DC": ["SynthCluster", "SynthTrace"]}
PROBLEM_POOL = [
    "it does not handle {a} under distribution shift",
    "evaluation on {b} in low-resource settings remains an open question",
    "the approach is limited to English data and its behaviour on other languages has not been explored",
    "the computational cost of {a} prevents deployment on edge devices",
    "it remains unclear whether {a} transfers to {b}",
]
NARROW_METHOD = "sequence"          # always evaluated on a single dataset (evaluation gap)
NEVER_PAIR = ("contrastive", "object detection")   # (method, task) never co-occur (combinatorial gap)


def _rng(arxiv_id: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(arxiv_id.encode()).hexdigest()[:8], 16))


def synthetic_sections(arxiv_id: str, title: str, abstract: str, category: str) -> List[Section]:
    rng = _rng(arxiv_id)
    words = [w.strip(",.:").lower() for w in (title + " " + abstract).split()]
    a = title.split(":")[0].strip().lower() or "the method"
    b = next((w for w in words[::-1] if len(w) > 6), "new domains")
    datasets = DATASETS.get(category, ["SynthBench"])
    used = ([datasets[0]] if NARROW_METHOD in title.lower()
            else [str(x) for x in rng.choice(datasets, size=min(2, len(datasets)), replace=False)])
    problems = [str(p) for p in rng.choice(PROBLEM_POOL, size=2, replace=False)]
    lim = " ".join(f"A limitation of our approach is that {p.format(a=a, b=b)}." for p in problems[:1])
    fut = (f"Future work should explore {b} for {category} research. "
           f"It remains an open question whether {problems[1].format(a=a, b=b)}.")
    return [
        Section(0, None, "Abstract", "abstract", abstract, "abstract"),
        Section(1, None, "Introduction", "introduction", f"{abstract} We address {a}.", "S1"),
        Section(2, None, "Experiments", "experiments",
                f"We evaluate {a} on {', '.join(used)} and report accuracy and F1.", "S4"),
        Section(3, None, "Limitations", "limitations", lim, "S6"),
        Section(4, None, "Conclusion and Future Work", "future_work", fut, "S7"),
    ]
