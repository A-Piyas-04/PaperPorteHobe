import { useState } from "react";
import type { Area, Payload } from "./types";
import { arxivUrl, authorList, fmtDate, fmtInt, pdfUrl, shortAuthors } from "./util";

interface Props {
  data: Payload;
  selected: number | null;
  areaById: Map<number, Area>;
  areaFilter: number | null;
  degree: number[];
  onSelect: (i: number) => void;
  onHover: (i: number | null) => void;
  onArea: (id: number | null) => void;
  onOpenArea: (id: number) => void;
}

const MAX_AUTHORS = 6;

function Overview({ data, areaFilter, onArea }: Props) {
  const totalCites = data.papers.reduce((n, p) => n + p.ci, 0);
  return (
    <div className="detail fade-in">
      <div className="eyebrow">Overview</div>
      <h2 className="detail-title">Your next read starts here</h2>
      <p>Choose a title from the paper list. Read its abstract, open the original on arXiv, then follow related papers to understand the area.</p>
      <div className="stats">
        <div><b>{fmtInt(data.papers.length)}</b><span>papers</span></div>
        <div><b>{data.areas.length}</b><span>areas</span></div>
        <div><b>{fmtInt(totalCites)}</b><span>citations</span></div>
      </div>
      <div className="section-title">Areas</div>
      <div className="area-list">
        {data.areas.map((a) => (
          <button
            key={a.id}
            className={`area-row${areaFilter === a.id ? " active" : ""}`}
            onClick={() => onArea(areaFilter === a.id ? null : a.id)}
          >
            <span className="dot" style={{ background: a.color }} />
            <span className="area-name">{a.name}</span>
            <span className="area-count">{a.size}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function DetailPane(props: Props) {
  const { data, selected } = props;
  const [expanded, setExpanded] = useState(false);
  const [lastSel, setLastSel] = useState<number | null>(null);
  if (selected !== lastSel) {
    setLastSel(selected);
    setExpanded(false);
  }

  if (selected === null) {
    return <div className="pane detail-pane"><Overview {...props} /></div>;
  }

  const p = data.papers[selected];
  const area = props.areaById.get(p.c);
  const authors = authorList(p);
  const nbrs = data.neighbors[String(selected)] ?? [];
  const longAbstract = p.ab.length > 420;

  return (
    <div className="pane detail-pane">
      <div className="detail fade-in" key={selected}>
        {area && (
          <button className="area-chip" onClick={() => props.onArea(area.id)} title="Show only this area">
            <span className="dot" style={{ background: area.color }} />
            {area.name}
          </button>
        )}
        <h2 className="detail-title">{p.t}</h2>
        <div className="authors">
          {authors.length === 0 ? "Unknown authors" : authors.slice(0, MAX_AUTHORS).join(", ")}
          {authors.length > MAX_AUTHORS && <span className="muted"> +{authors.length - MAX_AUTHORS} more</span>}
        </div>
        <div className="meta-line">
          <span>{fmtDate(p.d)}</span>
          <span className="dotsep">·</span>
          <span>{p.cat}</span>
          {p.v && (<><span className="dotsep">·</span><span>{p.v}</span></>)}
        </div>

        <div className="stats">
          <div><b>{fmtInt(p.ci)}</b><span>citations</span></div>
          <div><b>{fmtInt(p.rf)}</b><span>references</span></div>
          <div><b>{props.degree[selected]}</b><span>similar</span></div>
        </div>

        <div className={`abstract${expanded || !longAbstract ? " open" : ""}`}>{p.ab}</div>
        {longAbstract && (
          <button className="link-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? "Show less" : "Show more"}
          </button>
        )}

        <div className="actions">
          <a className="btn primary" href={arxivUrl(p.id)} target="_blank" rel="noopener noreferrer">Open on arXiv</a>
          <a className="btn" href={pdfUrl(p.id)} target="_blank" rel="noopener noreferrer">PDF</a>
        </div>

        {nbrs.length > 0 && (
          <>
            <div className="section-title">Continue with related papers</div>
            <p className="muted">Connections use text similarity and may include shared references. They do not establish a research gap.</p>
            <div className="similar">
              {nbrs.map(([j, s]) => {
                const q = data.papers[j];
                const qa = props.areaById.get(q.c);
                return (
                  <button
                    key={j}
                    className="sim-row"
                    onClick={() => props.onSelect(j)}
                    onMouseEnter={() => props.onHover(j)}
                    onMouseLeave={() => props.onHover(null)}
                  >
                    <span className="dot" style={{ background: qa?.color ?? "#cfd3da" }} />
                    <span className="sim-body">
                      <span className="sim-title">{q.t}</span>
                      <span className="sim-meta">{shortAuthors(q)} · similarity score {s.toFixed(2)}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </>
        )}

        {area && (
          <button className="link-btn area-open" onClick={() => props.onOpenArea(area.id)}>
            Open the {area.name} area page →
          </button>
        )}
      </div>
    </div>
  );
}
