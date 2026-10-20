"""Plain-language definitions for research jargon, with inline tooltips (<abbr>) for page text."""
from __future__ import annotations

import html

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


def definition(word: str) -> str:
    w = word.lower().strip()
    if w in GLOSSARY:
        return GLOSSARY[w]
    for k, v in GLOSSARY.items():
        if k.startswith(w) or k.split(" (")[0] == w or (w.endswith("s") and k == w[:-1]):
            return v
    return ""


def term(word: str) -> str:
    """Inline tooltip: <abbr title='...'>word</abbr> (screen readers read the title)."""
    d = definition(word)
    if not d:
        return html.escape(word)
    return f"<abbr class='sg-term' title='{html.escape(d, quote=True)}'>{html.escape(word)}</abbr>"


def render_list() -> str:
    return "".join(f"<p><b>{html.escape(k)}</b>: {html.escape(v)}</p>" for k, v in GLOSSARY.items())
