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
  onArea: (id: number) => void;
  onOpenArea: (id: number) => void;
  savedIds: string[];
  onSave: (id: string) => void;
  showFocusConnections?: boolean;
  onFocusConnections?: () => void;
}

const MAX_AUTHORS = 6;

function Overview({ areaFilter, areaById }: Props) {
  const area = areaFilter === null ? null : areaById.get(areaFilter);
  return (
    <div className="detail">
      <div className="eyebrow">Read with context</div>
      <h2 className="detail-title">Your next read starts here</h2>
      <p>{area ? `You are browsing ${area.name}.` : "You are browsing papers from the current dataset."}</p>
      <ol className="reading-guide">
        <li><b>Choose a title</b><p>Open a paper from the list to read its abstract here.</p></li>
        <li><b>Follow related work</b><p>Discover papers with similar ideas, one connection at a time.</p></li>
        <li><b>Keep what is useful</b><p>Save papers to your reading list and export your references before leaving.</p></li>
      </ol>
      <p className="muted">Your selection stays here until you choose another paper.</p>
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

        <div className="section-title">Abstract excerpt</div>
        <div className={`abstract${expanded || !longAbstract ? " open" : ""}`}>{p.ab || "No abstract available. Open the original paper on arXiv."}</div>
        {longAbstract && (
          <button className="link-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? "Show less" : "Show more"}
          </button>
        )}

        <div className="actions">
          <a className="btn primary" href={arxivUrl(p.id)} target="_blank" rel="noopener noreferrer">Open on arXiv</a>
          <a className="btn" href={pdfUrl(p.id)} target="_blank" rel="noopener noreferrer">PDF</a>
        </div>
        <button className="save-paper" aria-pressed={props.savedIds.includes(p.id)} onClick={() => props.onSave(p.id)}>
          {props.savedIds.includes(p.id) ? "Remove from reading list" : "Save to reading list"}
        </button>
        <p className="muted">Saved for this session. Export from Reading list before leaving.</p>

        {props.showFocusConnections && nbrs.length > 0 && props.onFocusConnections && (
          <button className="focus-connections" onClick={props.onFocusConnections}>
            Focus this paper&rsquo;s connections on the map →
          </button>
        )}

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
