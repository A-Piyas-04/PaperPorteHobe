"""BibTeX export shared by the app (reading list, reading path) and the library (research plans)."""
from __future__ import annotations

from typing import Dict, Iterable, Optional


def paper_url(paper_id: str, url: Optional[str] = None) -> str:
    pid = str(paper_id or "")
    if url:
        return str(url)
    if pid.startswith("doi:"):
        return f"https://doi.org/{pid[4:]}"
    if not pid or ":" in pid:
        return ""
    if pid.startswith("W") and pid[1:].isdigit():
        return f"https://openalex.org/{pid}"
    return f"https://arxiv.org/abs/{pid}"


def to_bibtex(rows: Iterable[Dict]) -> str:
    out = []
    for r in rows:
        aid = str(r.get("arxiv_id") or "")
        doi = str(r.get("doi") or "")
        pid = str(r.get("paper_id") or aid)
        key = "".join(ch if ch.isalnum() else "_" for ch in (f"arxiv{aid}" if aid else pid))
        authors = " and ".join(a.strip() for a in str(r.get("authors") or "").split(";") if a.strip())
        fields = [("title", r.get("title", "")), ("author", authors), ("year", str(r.get("date") or "")[:4])]
        if aid:
            fields += [("eprint", aid), ("archivePrefix", "arXiv"),
                       ("primaryClass", r.get("primary_category") or "")]
        if doi:
            fields.append(("doi", doi))
        if r.get("venue") and not aid:
            fields.append(("howpublished", r["venue"]))
        url = paper_url(pid, r.get("url") or None) or (f"https://doi.org/{doi}" if doi else "")
        if url:
            fields.append(("url", url))
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
        out.append(f"@misc{{{key},\n{body}\n}}")
    return "\n\n".join(out) + "\n"
