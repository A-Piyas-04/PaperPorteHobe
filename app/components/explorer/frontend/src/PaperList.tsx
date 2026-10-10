import { useEffect, useRef, useState } from "react";
import type { Paper } from "./types";
import { fmtDate, shortAuthors } from "./util";

export interface ListItem {
  i: number;
  score?: number;
}

interface Props {
  title: string;
  items: ListItem[];
  papers: Paper[];
  colors: Map<number, string>;
  degree: number[];
  hovered: number | null;
  selected: number | null;
  loading: boolean;
  onHover: (i: number | null) => void;
  onSelect: (i: number) => void;
}

const PAGE = 20;

export default function PaperList(props: Props) {
  const [limit, setLimit] = useState(PAGE);
  const scrollRef = useRef<HTMLDivElement>(null);
  const shown = props.items.slice(0, limit);

  // Keep the selected paper in view when it is picked from the graph.
  useEffect(() => {
    if (props.selected === null) return;
    const el = scrollRef.current?.querySelector<HTMLElement>(`[data-i="${props.selected}"]`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [props.selected]);

  return (
    <div className="pane list-pane">
      <div className="pane-head">
        <span className="pane-title">{props.title}</span>
        <span className="pane-meta">{props.loading ? "" : props.items.length}</span>
      </div>
      <div className="list-scroll" ref={scrollRef}>
        {props.loading
          ? Array.from({ length: 8 }).map((_, k) => (
              <div key={k} className="skel-row"><div className="skel" /><div className="skel short" /></div>
            ))
          : shown.map((it, k) => {
              const p = props.papers[it.i];
              const cls = ["row"];
              if (it.i === props.selected) cls.push("selected");
              if (it.i === props.hovered) cls.push("hovered");
              return (
                <button
                  key={it.i}
                  data-i={it.i}
                  className={cls.join(" ")}
                  aria-pressed={it.i === props.selected}
                  style={{ animationDelay: `${Math.min(k, 14) * 22}ms`, borderLeftColor: props.colors.get(p.c) ?? "#cfd3da" }}
                  onMouseEnter={() => props.onHover(it.i)}
                  onMouseLeave={() => props.onHover(null)}
                  onClick={() => props.onSelect(it.i)}
                >
                  <span className="row-title">{p.t}</span>
                  <span className="row-meta">
                    <span>{shortAuthors(p)}</span>
                    <span className="dotsep">·</span>
                    <span>{fmtDate(p.d)}</span>
                    <span className="row-right">
                      {it.score !== undefined
                        ? <span className="match">Rank {k + 1}</span>
                        : p.ci > 0
                          ? <span>{p.ci} cites</span>
                          : <span>{props.degree[it.i]} links</span>}
                    </span>
                  </span>
                </button>
              );
            })}
        {!props.loading && props.items.length === 0 && (
          <div className="empty">No papers match these filters.</div>
        )}
        {!props.loading && props.items.length > limit && (
          <button className="more" onClick={() => setLimit(limit + PAGE)}>
            Show {Math.min(PAGE, props.items.length - limit)} more
          </button>
        )}
      </div>
    </div>
  );
}
