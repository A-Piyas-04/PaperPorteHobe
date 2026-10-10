import { useEffect, useMemo, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import TopBar from "./TopBar";
import PaperList from "./PaperList";
import DetailPane from "./DetailPane";
import MapView, { type CameraCmd } from "./MapView";
import type { Args, Scope, UIEvent } from "./types";

type Outgoing = UIEvent extends infer E ? (E extends UIEvent ? Omit<E, "nonce"> : never) : never;
function send(event: Outgoing) {
  const nonce = Date.now();
  Streamlit.setComponentValue({ ...event, nonce });
  return nonce;
}

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export default function App({ args }: { args: Args | null }) {
  const data = args?.data;
  const hl = args?.highlight;

  const [view, setView] = useState<"map" | "list">("map");
  const [selected, setSelected] = useState<number | null>(null);
  const [scope, setScope] = useState<Scope>({ kind: "overview" });
  const historyRef = useRef<Scope[]>([]);
  const [cameraCmd, setCameraCmd] = useState<CameraCmd | null>(null);
  const [focusNonce, setFocusNonce] = useState(0);
  const [searchNonce, setSearchNonce] = useState(0);

  const [areaFilter, setArea] = useState<number | null>(null);
  const [catFilter, setCategory] = useState<string | null>(null);
  const [sort, setSort] = useState("newest");
  const [areaQuery, setAreaQuery] = useState("");
  const [mobileDetail, setMobileDetail] = useState(false);
  const [pending, setPending] = useState<number | null>(null);
  const [timedOut, setTimedOut] = useState(false);

  const colors = useMemo(() => new Map((data?.areas ?? []).map((a) => [a.id, a.color])), [data]);
  const areaById = useMemo(() => new Map((data?.areas ?? []).map((a) => [a.id, a])), [data]);
  const degree = useMemo(() => {
    const counts = new Array(data?.papers.length ?? 0).fill(0);
    for (const [a, b] of data?.edges ?? []) { counts[a]++; counts[b]++; }
    return counts;
  }, [data]);
  const reducedMotion = useMemo(prefersReducedMotion, []);

  useEffect(() => { Streamlit.setFrameHeight(960); }, [args]);

  // A new search result set: frame it on the map and reset the scope history.
  useEffect(() => {
    setPending(null);
    setTimedOut(false);
    setSelected(null);
    setMobileDetail(false);
    if (hl) {
      historyRef.current = [];
      setScope({ kind: "search" });
      setSearchNonce((n) => n + 1);
      setArea(null);
      setCategory(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hl?.query, hl?.req]);

  // Open a research area sent from the Areas page.
  useEffect(() => {
    if (args?.focus_area != null) {
      historyRef.current = [{ kind: "overview" }];
      setScope({ kind: "area", areaId: args.focus_area });
      setArea(args.focus_area);
      setSelected(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args?.focus_area]);

  // Preselect a specific paper (index-zero contract / deep link).
  useEffect(() => {
    if (args?.select != null && data?.papers[args.select]) {
      historyRef.current = [{ kind: "overview" }];
      setScope({ kind: "paper", root: args.select, depth: 1 });
      setSelected(args.select);
      setMobileDetail(true);
      setFocusNonce((n) => n + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args?.select, data]);

  useEffect(() => {
    if (pending === null) return;
    const timer = window.setTimeout(() => { setPending(null); setTimedOut(true); }, 45000);
    return () => window.clearTimeout(timer);
  }, [pending]);

  if (!data) return <div role="status" className="empty">Loading your research workspace…</div>;

  const searchItems = hl ? hl.items.map(([i]) => i) : [];

  // ---- scope + selection transitions -----------------------------------
  const goScope = (next: Scope, pushPrev = true) => {
    if (pushPrev) historyRef.current.push(scope);
    setScope(next);
  };
  const focusArea = (id: number) => {
    setSelected(null);
    setMobileDetail(false);
    setArea(id);
    goScope({ kind: "area", areaId: id });
  };
  const selectPaper = (i: number, opts: { external?: boolean; reframe?: boolean } = {}) => {
    setSelected(i);
    setMobileDetail(true);
    if (opts.reframe && view === "map") goScope({ kind: "paper", root: i, depth: 1 });
    if (opts.external) setFocusNonce((n) => n + 1);
  };
  const focusConnections = () => { if (selected !== null) goScope({ kind: "paper", root: selected, depth: 1 }); };
  const expand = () => { if (scope.kind === "paper") setScope({ ...scope, depth: scope.depth + 1 }); };
  const back = () => {
    const prev = historyRef.current.pop();
    if (prev) { setScope(prev); setSelected(null); setMobileDetail(false); }
  };
  const toOverview = () => {
    historyRef.current = [];
    setScope({ kind: "overview" });
    setSelected(null);
    setArea(null);
    setMobileDetail(false);
  };
  const clearSelection = () => { setSelected(null); setMobileDetail(false); };
  const camera = (kind: CameraCmd["kind"]) => setCameraCmd({ kind, n: Date.now() });

  // ---- list-view item set ----------------------------------------------
  const visible = (i: number) => {
    const p = data.papers[i];
    return p && (areaFilter === null || p.c === areaFilter) && (catFilter === null || p.cat === catFilter);
  };
  const items = hl
    ? hl.items.filter(([i]) => visible(i)).map(([i, score]) => ({ i, score }))
    : data.papers.filter((p) => visible(p.i)).sort((a, b) => sort === "cited"
      ? b.ci - a.ci || b.d.localeCompare(a.d) : b.d.localeCompare(a.d) || a.t.localeCompare(b.t)).map((p) => ({ i: p.i }));
  const areas = data.areas.filter((a) => a.name.toLowerCase().includes(areaQuery.toLowerCase()));

  // ---- map scope summary -----------------------------------------------
  const scopeArea = scope.kind === "area" ? areaById.get(scope.areaId) : null;
  const sampled = data.renderedTotal < data.corpusTotal;

  const crumbs = (
    <nav className="map-crumbs" aria-label="Map location">
      <button className={`crumb${scope.kind === "overview" ? " current" : ""}`} onClick={toOverview}>Overview</button>
      {scope.kind === "search" && <><span className="sep">›</span><span className="crumb current">Search results</span></>}
      {scopeArea && <><span className="sep">›</span><span className="crumb current">{scopeArea.name}</span></>}
      {scope.kind === "paper" && <><span className="sep">›</span><span className="crumb current">This paper&rsquo;s connections</span></>}
      {selected !== null && <><span className="sep">›</span><span className="crumb current selected-crumb">{data.papers[selected].t}</span></>}
    </nav>
  );

  const legend = (
    <div className="map-legend">
      <div className="legend-title">How to read this map</div>
      <ul>
        <li><span className="lg-swatch" style={{ background: "#2f6feb" }} /><span className="lg-swatch" style={{ background: "#e8710a" }} /><span className="lg-swatch" style={{ background: "#1a9e6c" }} /> Colour groups papers into research areas.</li>
        <li><span className="lg-size" /> Node size reflects {data.sizeBy === "citations" ? "citations" : "number of connections"}.</li>
        <li><span className="lg-edge" /> A link means two papers are textually similar (and may share references). It is not a citation or a confidence score.</li>
      </ul>
      <p className="muted">Position comes from an embedding projection; screen distance is a rough guide to similarity, not a measure of research value or gaps.</p>
    </div>
  );

  const detailDrawer = (
    <aside className={`map-drawer${mobileDetail && selected !== null ? " open-mobile" : ""}`} aria-label="Paper details">
      {selected !== null && (
        <button className="drawer-close" onClick={() => setMobileDetail(false)}>← Back to map</button>
      )}
      {selected === null ? (
        <div className="drawer-intro">
          <div className="eyebrow">
            {scope.kind === "overview" ? "The research landscape"
              : scope.kind === "search" ? "Search results on the map"
              : scopeArea ? scopeArea.name : "Paper connections"}
          </div>
          <h2 className="detail-title">
            {scope.kind === "overview" ? "Start from an area"
              : scope.kind === "search" ? `${searchItems.length} papers to explore`
              : "Pick a paper to read"}
          </h2>
          <p className="muted">
            {scope.kind === "overview"
              ? "Each bubble is a research area. Click one to reveal its papers and how they connect."
              : "Click any node to read its abstract, metadata and related work here. Your choice stays selected until you pick another."}
          </p>
          {legend}
        </div>
      ) : (
        <DetailPane
          data={data} selected={selected} areaById={areaById} areaFilter={areaFilter} degree={degree}
          savedIds={args?.saved_ids ?? []} onSave={(id) => send({ type: "save", id })}
          onSelect={(j) => selectPaper(j, { external: true, reframe: true })}
          onHover={() => {}} onArea={focusArea} onOpenArea={(id) => send({ type: "open_area", id })}
          showFocusConnections={scope.kind !== "paper"} onFocusConnections={focusConnections} />
      )}
    </aside>
  );

  return (
    <main className="shell workspace">
      <header className="workspace-heading">
        <div className="eyebrow">Your research workspace</div>
        <h1>See the landscape. Follow the connections.</h1>
        <p>Start from the map to see how research areas relate, or switch to the list to read and filter papers.</p>
      </header>

      <TopBar data={data} query={hl?.query ?? ""} matches={items.length} pending={pending !== null}
        areaFilter={areaFilter} catFilter={catFilter} visibleCount={items.length}
        onSearch={(q) => { setTimedOut(false); setPending(send({ type: "search", q })); }}
        onClear={() => { setPending(null); setSelected(null); send({ type: "clear" }); toOverview(); }}
        onArea={(id) => (id === null ? toOverview() : focusArea(id))}
        onCategory={(cat) => { setCategory(cat); setSelected(null); setView("list"); }}
        showFilters={view === "list"} />

      {timedOut && <div role="alert" className="notice">Search is taking longer than expected. Try searching again or return to Find papers.</div>}

      <nav className="workspace-tabs" aria-label="Workspace view">
        <button aria-pressed={view === "map"} onClick={() => setView("map")}>🗺️ Map</button>
        <button aria-pressed={view === "list"} onClick={() => setView("list")}>
          ☰ List{hl ? ` · ${items.length}` : ""}
        </button>
      </nav>

      {view === "map" ? (
        <section className="map-view" aria-label="Research map">
          <div className="map-bar">
            {crumbs}
            <div className="map-controls">
              {historyRef.current.length > 0 && <button className="mc" onClick={back} title="Back">← Back</button>}
              {selected !== null && <button className="mc" onClick={clearSelection} title="Clear selection">Clear selection</button>}
              {scope.kind === "paper" && <button className="mc" onClick={expand} title="Show more connected papers">+ Expand</button>}
              <span className="mc-group">
                <button className="mc icon" onClick={() => camera("zoomIn")} aria-label="Zoom in">+</button>
                <button className="mc icon" onClick={() => camera("zoomOut")} aria-label="Zoom out">−</button>
                <button className="mc" onClick={() => camera("fit")} aria-label="Fit view">Fit</button>
              </span>
              <button className="mc" onClick={toOverview} title="Return to the overview">Overview</button>
            </div>
          </div>
          {sampled && (
            <p className="map-sample muted">
              Showing {data.renderedTotal.toLocaleString()} of {data.corpusTotal.toLocaleString()} papers
              (most-cited per area). Search covers every paper; matches are always added to the map and List.
            </p>
          )}
          {scope.kind === "search" && searchItems.length === 0 && (
            <div className="notice" role="status">No papers matched this search, so the map has nothing to show. Try broader words, or return to the overview.</div>
          )}
          {scope.kind === "paper" && (data.neighbors[String(scope.root)]?.length ?? 0) === 0 && (
            <div className="notice" role="status">This paper has no stored connections. Open it in the List to read it, or return to its area.</div>
          )}
          <div className="map-stage">
            <MapView data={data} colors={colors} scope={scope} searchNonce={searchNonce} searchItems={searchItems}
              selected={selected} focusNonce={focusNonce} cameraCmd={cameraCmd} reducedMotion={reducedMotion}
              onSelectPaper={(i) => selectPaper(i)} onFocusArea={focusArea} onClearSelection={clearSelection} />
            {detailDrawer}
          </div>
        </section>
      ) : (
        <section className="list-view">
          <div className="reading-context">
            <div>
              <b>{hl ? `Results for “${hl.query}”` : areaFilter !== null ? areaById.get(areaFilter)?.name : "All papers"}</b>
              <p>{hl ? "Results are ranked by relevance. Choose a paper to read its abstract and related work."
                : "Choose a title to read its abstract. Related papers stay visible until you choose another."}</p>
            </div>
            {!hl && (
              <label>Order <select aria-label="Paper order" value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="newest">Newest first</option><option value="cited">Most cited</option>
              </select></label>
            )}
          </div>
          {!areaFilter && !hl && (
            <div className="area-browser">
              <div className="browser-head">
                <div><h2>Browse research areas</h2><p>These topics group papers with similar titles and abstracts. Open one to start reading, or jump to the Map.</p></div>
                <input aria-label="Find a research area" placeholder="Filter areas by name…" value={areaQuery} onChange={(e) => setAreaQuery(e.target.value)} />
              </div>
              <div className="area-grid">{areas.map((a) => (
                <button className="topic-card" key={a.id} onClick={() => focusArea(a.id)}>
                  <span className="topic-count"><span className="dot" style={{ background: a.color }} /> {a.size.toLocaleString()} papers in corpus</span>
                  <h3>{a.name}</h3><span className="topic-action">Explore papers →</span>
                </button>))}
              </div>
              {!areas.length && <div className="empty">No areas with that name. Try a broader term or browse all papers.</div>}
            </div>
          )}
          {(areaFilter !== null || hl) && (
            <>
              {hl?.ambiguous && <div className="notice">This query spans several areas. Use the area filter to narrow your interest.</div>}
              {!items.length && pending === null && <div className="notice" role="status">No papers to show. Clear the area or category filters, or try a broader search.</div>}
              <div className={`reading-panes${mobileDetail && selected !== null ? " show-detail" : ""}`}>
                <PaperList key={`${hl?.req}|${hl?.query}|${areaFilter}|${catFilter}|${sort}`} title="Choose a paper" items={items}
                  papers={data.papers} colors={colors} degree={degree} hovered={null} selected={selected}
                  loading={pending !== null} onHover={() => {}} onSelect={(i) => selectPaper(i)} />
                <section className="reading-detail" aria-label="Paper details">
                  {selected !== null && <button className="back-to-list" onClick={() => setMobileDetail(false)}>← Back to papers</button>}
                  <DetailPane data={data} selected={selected} areaById={areaById} areaFilter={areaFilter} degree={degree}
                    savedIds={args?.saved_ids ?? []} onSave={(id) => send({ type: "save", id })}
                    onSelect={(j) => selectPaper(j)} onHover={() => {}} onArea={focusArea}
                    onOpenArea={(id) => send({ type: "open_area", id })} />
                </section>
              </div>
            </>
          )}
        </section>
      )}
    </main>
  );
}
