import { useEffect, useState } from "react";
import type { Payload } from "./types";
import { fmtInt } from "./util";

interface Props {
  data: Payload;
  query: string;
  matches: number;
  pending: boolean;
  areaFilter: number | null;
  catFilter: string | null;
  visibleCount: number;
  onSearch: (q: string) => void;
  onClear: () => void;
  onArea: (id: number | null) => void;
  onCategory: (cat: string | null) => void;
}

export default function TopBar(props: Props) {
  const { data } = props;
  const [q, setQ] = useState(props.query);
  useEffect(() => setQ(props.query), [props.query]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = q.trim();
    if (text) props.onSearch(text);
  };

  const filtered = props.areaFilter !== null || props.catFilter !== null;

  return (
    <div className="topbar">
      <form className="search" onSubmit={submit}>
        <svg className="search-icon" viewBox="0 0 24 24" width="20" height="20" aria-hidden>
          <path d="M10.5 3a7.5 7.5 0 0 1 5.9 12.13l4.24 4.24-1.41 1.41-4.24-4.24A7.5 7.5 0 1 1 10.5 3Zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z" fill="currentColor" />
        </svg>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search a topic, method or problem"
          aria-label="Search papers"
        />
        {props.query && (
          <button type="button" className="ghost" onClick={() => { setQ(""); props.onClear(); }}>
            Clear
          </button>
        )}
        <button type="submit" className="primary" disabled={props.pending || !q.trim()}>
          {props.pending ? <span className="btn-spin" /> : "Search"}
        </button>
      </form>

      <div className="filters">
        <select
          value={props.areaFilter ?? ""}
          onChange={(e) => props.onArea(e.target.value === "" ? null : Number(e.target.value))}
          aria-label="Filter by research area"
        >
          <option value="">All areas</option>
          {data.areas.map((a) => (
            <option key={a.id} value={a.id}>{a.name} ({a.size})</option>
          ))}
        </select>
        <select
          value={props.catFilter ?? ""}
          onChange={(e) => props.onCategory(e.target.value === "" ? null : e.target.value)}
          aria-label="Filter by arXiv category"
        >
          <option value="">All categories</option>
          {data.categories.map(([c, n]) => (
            <option key={c} value={c}>{c} ({n})</option>
          ))}
        </select>
        {filtered && (
          <button className="ghost" onClick={() => { props.onArea(null); props.onCategory(null); }}>
            Reset
          </button>
        )}
      </div>

      <div className="count">
        {props.query && !props.pending ? (
          <span><b>{props.matches}</b> matches</span>
        ) : (
          <span><b>{fmtInt(props.visibleCount)}</b> papers</span>
        )}
        <span className="sep">·</span>
        <span>{fmtInt(data.edges.length)} links</span>
      </div>
    </div>
  );
}
