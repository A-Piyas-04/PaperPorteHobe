"""Is this paper a survey / review / tutorial? Cheap and transparent; run at load time too."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TITLE = re.compile(r"\b(survey|review|overview|tutorial|primer|systematic\s+(literature\s+)?review|"
                    r"sok\b|systemati[sz]ation\s+of\s+knowledge|taxonomy|roadmap|state[- ]of[- ]the[- ]art\s+in|"
                    r"a\s+decade\s+of|lessons\s+learned|perspectives?\s+on)\b", re.I)
_ABSTRACT = re.compile(r"^(this|in\s+this)\s+(survey|review|tutorial)|\bwe\s+(survey|review)\s+(the\s+)?"
                       r"(recent\s+)?(literature|work|advances|progress|methods)", re.I)
_NEG = re.compile(r"\bpeer[- ]review|review\s+(generation|comments?|summari[sz]ation)|code\s+review|"
                  r"product\s+review|reviewer", re.I)


def is_survey(title: str, abstract: str = "", openalex_type: str = "") -> bool:
    if openalex_type == "review":
        return True
    t = str(title or "")
    if _NEG.search(t):
        return False
    return bool(_TITLE.search(t) or _ABSTRACT.search(str(abstract or "")[:400]))


def survey_mask(df: pd.DataFrame) -> np.ndarray:
    return np.array([is_survey(t, a) for t, a in zip(df["title"].astype(str), df["abstract"].astype(str))],
                    dtype=bool)
