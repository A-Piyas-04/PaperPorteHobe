import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import GraphView, { type CameraCmd } from "./GraphView";
import TopBar from "./TopBar";
import PaperList, { type ListItem } from "./PaperList";
import DetailPane from "./DetailPane";
import type { Args, UIEvent } from "./types";

const NAV_OFFSET = 112;

function useFrameHeight(fallback: number) {
  const [h, setH] = useState(fallback);
  useEffect(() => {
    let parentWin: Window | null = null;
    const calc = () => {
      try {
        const ph = window.parent.innerHeight;
        setH(Math.max(620, ph - NAV_OFFSET));
      } catch {
        setH(fallback);
      }
    };
    calc();
    try {
      parentWin = window.parent;
      parentWin.addEventListener("resize", calc);
    } catch {
      parentWin = null;
    }
    return () => parentWin?.removeEventListener("resize", calc);
  }, [fallback]);
  useEffect(() => {
    Streamlit.setFrameHeight(h);
  }, [h]);
  return h;
}

type Outgoing = UIEvent extends infer E ? (E extends UIEvent ? Omit<E, "nonce"> : never) : never;

function send(ev: Outgoing) {
  const nonce = Date.now();
  Streamlit.setComponentValue({ ...ev, nonce });
  return nonce;
}

function Skeleton({ height }: { height: number }) {
  return (
    <div className="shell" style={{ height }}>
      <div className="topbar"><div className="skel" style={{ width: 420, height: 44 }} /></div>
      <div className="panes">
        <div className="pane list-pane">
          {Array.from({ length: 8 }).map((_, k) => <div key={k} className="skel-row"><div className="skel" /><div className="skel short" /></div>)}
        </div>
        <div className="pane graph-pane"><div className="graph-loading"><div className="spinner" /></div></div>
        <div className="pane detail-pane"><div className="skel" style={{ height: 28, width: "70%" }} /></div>
      </div>
    </div>
  );
}

export default function App({ args }: { args: Args | null }) {
  const height = useFrameHeight(args?.height ?? 780);
  const data = args?.data ?? null;

  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [areaFilter, setAreaFilter] = useState<number | null>(null);
  const [catFilter, setCatFilter] = useState<string | null>(null);
  const [camera, setCamera] = useState<CameraCmd | null>(null);
  const [pending, setPending] = useState<number | null>(null);
  const camN = useRef(0);
  const moveCamera = useCallback((cmd: Omit<CameraCmd, "n"> & Record<string, unknown>) => {
    camN.current += 1;
    setCamera({ ...(cmd as CameraCmd), n: camN.current });
  }, []);

  const colors = useMemo(() => new Map((data?.areas ?? []).map((a) => [a.id, a.color])), [data]);
  const areaById = useMemo(() => new Map((data?.areas ?? []).map((a) => [a.id, a])), [data]);
  const degree = useMemo(() => {
    const d = new Array(data?.papers.length ?? 0).fill(0);
    for (const [a, b] of data?.edges ?? []) { d[a] += 1; d[b] += 1; }
    return d as number[];
  }, [data]);

  const hl = args?.highlight ?? null;
  const highlight = useMemo(() => {
    if (!hl || !hl.items.length) return null;
    return new Map(hl.items.map(([i], rank) => [i, rank]));
  }, [hl]);

  const visible = useCallback(
    (i: number) => {
      const p = data?.papers[i];
      if (!p) return false;
      return (areaFilter === null || p.c === areaFilter) && (catFilter === null || p.cat === catFilter);
    },
    [data, areaFilter, catFilter],
  );

  // New search results arrive from Python: clear pending, frame the matches.
  const hlKey = hl ? `${hl.query}|${hl.req ?? ""}|${hl.items.length}` : "";
  useEffect(() => {
    setPending(null);
    if (hl && hl.items.length) {
      setSelected(null);
      moveCamera({ kind: "nodes", ids: hl.items.slice(0, 15).map(([i]) => i) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hlKey]);

  // Area focus requested by the Streamlit page.
  const focusArg = args?.focus_area ?? null;
  useEffect(() => {
    if (focusArg === null || !data) return;
    setAreaFilter(focusArg);
    setSelected(null);
    moveCamera({ kind: "nodes", ids: data.papers.filter((p) => p.c === focusArg).map((p) => p.i) });
  }, [focusArg, data, moveCamera]);

  // Paper preselected by the Streamlit page.
  const selectArg = args?.select ?? null;
  useEffect(() => {
    if (!selectArg || !data) return;
    const p = data.papers.find((q) => q.id === selectArg);
    if (p) {
      setSelected(p.i);
      moveCamera({ kind: "node", i: p.i, zoom: true });
    }
  }, [selectArg, data, moveCamera]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const items: ListItem[] = useMemo(() => {
    if (!data) return [];
    if (hl && hl.items.length) {
      return hl.items.filter(([i]) => visible(i)).map(([i, s]) => ({ i, score: s }));
    }
    return data.papers
      .filter((p) => visible(p.i))
      .sort((a, b) => b.ci - a.ci || degree[b.i] - degree[a.i])
      .map((p) => ({ i: p.i }));
  }, [data, hl, visible, degree]);

  if (!data) return <Skeleton height={height} />;

  const selectPaper = (i: number | null, fromGraph = false) => {
    setSelected(i);
    if (i !== null) moveCamera({ kind: "node", i, zoom: !fromGraph });
  };

  const focusArea = (id: number | null) => {
    setAreaFilter(id);
    setSelected(null);
    if (id === null) moveCamera({ kind: "reset" });
    else moveCamera({ kind: "nodes", ids: data.papers.filter((p) => p.c === id).map((p) => p.i) });
  };

  const setCategory = (cat: string | null) => {
    setCatFilter(cat);
    setSelected(null);
    if (cat === null) moveCamera({ kind: "reset" });
    else moveCamera({ kind: "nodes", ids: data.papers.filter((p) => p.cat === cat).map((p) => p.i) });
  };

  const visibleCount = data.papers.reduce((n, p) => n + (visible(p.i) ? 1 : 0), 0);
  const listTitle = hl && hl.items.length
    ? "Best matches"
    : areaFilter !== null
      ? areaById.get(areaFilter)?.name ?? "Area"
      : catFilter ?? "Most connected papers";

  return (
    <div className="shell" style={{ height }}>
      <TopBar
        data={data}
        query={hl?.query ?? ""}
        matches={hl?.items.length ?? 0}
        pending={pending !== null}
        areaFilter={areaFilter}
        catFilter={catFilter}
        visibleCount={visibleCount}
        onSearch={(q) => setPending(send({ type: "search", q }))}
        onClear={() => {
          setPending(null);
          send({ type: "clear" });
          moveCamera({ kind: "reset" });
        }}
        onArea={focusArea}
        onCategory={setCategory}
      />
      <div className="panes">
        <PaperList
          key={`${hlKey}|${areaFilter}|${catFilter}`}
          title={listTitle}
          items={items}
          papers={data.papers}
          colors={colors}
          degree={degree}
          hovered={hovered}
          selected={selected}
          loading={pending !== null}
          onHover={setHovered}
          onSelect={(i) => selectPaper(i)}
        />
        <div className="pane graph-pane">
          <GraphView
            data={data}
            colors={colors}
            hovered={hovered}
            selected={selected}
            highlight={highlight}
            areaFilter={areaFilter}
            catFilter={catFilter}
            camera={camera}
            onHover={setHovered}
            onSelect={(i) => selectPaper(i, true)}
            onArea={(id) => focusArea(id)}
          />
          <div className="graph-tools">
            <button className="tool" title="Fit to view" onClick={() => moveCamera({ kind: "reset" })}>Fit</button>
          </div>
          <div className="graph-legend">
            {data.sizeBy === "citations" ? "Node size = citations" : "Node size = number of similar papers"} · lines link similar papers
          </div>
        </div>
        <DetailPane
          data={data}
          selected={selected}
          areaById={areaById}
          areaFilter={areaFilter}
          degree={degree}
          onSelect={(i) => selectPaper(i)}
          onHover={setHovered}
          onArea={focusArea}
          onOpenArea={(id) => send({ type: "open_area", id })}
        />
      </div>
    </div>
  );
}
