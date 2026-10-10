import { useEffect, useMemo, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import TopBar from "./TopBar";
import PaperList from "./PaperList";
import DetailPane from "./DetailPane";
import type { Args, UIEvent } from "./types";

type Outgoing = UIEvent extends infer E ? (E extends UIEvent ? Omit<E, "nonce"> : never) : never;
function send(event: Outgoing) {
  const nonce = Date.now();
  Streamlit.setComponentValue({ ...event, nonce });
  return nonce;
}
export default function App({ args }: { args: Args | null }) {
  const data = args?.data;
  const hl = args?.highlight;
  const [selected, setSelected] = useState<number | null>(null);
  const [areaFilter, setArea] = useState<number | null>(null);
  const [catFilter, setCategory] = useState<string | null>(null);
  const [pending, setPending] = useState<number | null>(null);
  const [timedOut, setTimedOut] = useState(false);
  const [view, setView] = useState<"areas" | "papers">("areas");
  const [mobileDetail, setMobileDetail] = useState(false);
  const [areaQuery, setAreaQuery] = useState("");
  const [sort, setSort] = useState("newest");
  const colors = useMemo(() => new Map((data?.areas ?? []).map(a => [a.id, a.color])), [data]);
  const areaById = useMemo(() => new Map((data?.areas ?? []).map(a => [a.id, a])), [data]);
  const degree = useMemo(() => {
    const counts = new Array(data?.papers.length ?? 0).fill(0);
    for (const [a, b] of data?.edges ?? []) { counts[a]++; counts[b]++; }
    return counts;
  }, [data]);
  useEffect(() => { Streamlit.setFrameHeight(960); }, [args]);
  useEffect(() => {
    setPending(null); setTimedOut(false); setSelected(null);
    if (hl) { setView("papers"); setArea(null); setCategory(null); }
  }, [hl?.query, hl?.req]);
  useEffect(() => {
    if (args?.focus_area != null) { setArea(args.focus_area); setView("papers"); setSelected(null); }
  }, [args?.focus_area]);
  useEffect(() => {
    if (args?.select != null && data?.papers[args.select]) {
      setSelected(args.select); setView("papers"); setMobileDetail(true);
    }
  }, [args?.select, data]);
  useEffect(() => {
    if (pending === null) return;
    const timer = window.setTimeout(() => { setPending(null); setTimedOut(true); }, 45000);
    return () => window.clearTimeout(timer);
  }, [pending]);
  if (!data) return <div role="status" className="empty">Loading your research workspace…</div>;
  const select = (i: number) => { setSelected(i); setMobileDetail(true); };
  const focus = (id: number | null) => { setArea(id); setSelected(null); setMobileDetail(false); setView("papers"); };
  const visible = (i: number) => {
    const p = data.papers[i];
    return p && (areaFilter === null || p.c === areaFilter) && (catFilter === null || p.cat === catFilter);
  };
  const items = hl
    ? hl.items.filter(([i]) => visible(i)).map(([i, score]) => ({ i, score }))
    : data.papers.filter(p => visible(p.i)).sort((a, b) => sort === "cited"
      ? b.ci - a.ci || b.d.localeCompare(a.d) : b.d.localeCompare(a.d) || a.t.localeCompare(b.t)).map(p => ({ i: p.i }));
  const areas = data.areas.filter(a => a.name.toLowerCase().includes(areaQuery.toLowerCase()));
  return <main className="shell workspace">
    <header className="workspace-heading"><div className="eyebrow">Your research workspace</div>
      <h1>Explore one area. Follow one paper.</h1>
      <p>Choose an area, read a paper, then follow its closest connections.</p>
    </header>
    <TopBar data={data} query={hl?.query ?? ""} matches={items.length} pending={pending !== null}
      areaFilter={areaFilter} catFilter={catFilter} visibleCount={items.length}
      onSearch={q => { setTimedOut(false); setPending(send({ type: "search", q })); setView("papers"); }}
      onClear={() => { setPending(null); setSelected(null); send({ type: "clear" }); }}
      onArea={focus} onCategory={cat => { setCategory(cat); setSelected(null); setMobileDetail(false); setView("papers"); }} />
    {timedOut && <div role="alert" className="notice">Search is taking longer than expected. Try searching again or return to Find papers.</div>}
    <nav className="workspace-tabs" aria-label="Workspace view">
      <button aria-pressed={view === "areas"} onClick={() => setView("areas")}>Browse research areas</button>
      <button aria-pressed={view === "papers"} onClick={() => setView("papers")}>{hl ? "Search results" : "Browse papers"} · {items.length}</button>
    </nav>
    {view === "areas" ? <section className="area-browser">
      <div className="browser-head"><div><h2>Find your starting point</h2><p>These topics group papers with similar titles and abstracts. Open one to start reading.</p></div>
        <input aria-label="Find a research area" placeholder="Filter areas by name…" value={areaQuery} onChange={e => setAreaQuery(e.target.value)} /></div>
      <div className="area-grid">{areas.map(a => <button className="topic-card" key={a.id} onClick={() => { focus(a.id); if (hl) send({ type: "clear" }); }}>
        <span className="topic-count"><span className="dot" style={{ background: a.color }} /> {a.size.toLocaleString()} papers in corpus</span>
        <h3>{a.name}</h3><span className="topic-action">Explore papers →</span>
      </button>)}</div>
      {!areas.length && <div className="empty">No areas with that name. Try a broader term or browse all papers.</div>}
      <p className="muted">Some papers do not belong to a clear area. You can still find them in Browse papers.</p>
    </section> : <>
      <div className="reading-context"><div><b>{hl ? `Results for “${hl.query}”` : areaFilter !== null ? areaById.get(areaFilter)?.name : "All papers"}</b>
        <p>{hl ? "Results are ranked by relevance. Choose a paper to read its abstract and related work." : "Choose a title to read its abstract. Related papers stay visible until you choose another."}</p></div>
        {!hl && <label>Order <select aria-label="Paper order" value={sort} onChange={e => setSort(e.target.value)}><option value="newest">Newest first</option><option value="cited">Most cited</option></select></label>}
      </div>
      {hl?.ambiguous && <div className="notice">This query spans several areas. Use the area filter to narrow your interest.</div>}
      {!items.length && pending === null && <div className="notice" role="status">No papers to show. Clear the area or category filters, or try a broader search.</div>}
      <div className={`reading-panes${mobileDetail && selected !== null ? " show-detail" : ""}`}>
        <PaperList key={`${hl?.req}|${hl?.query}|${areaFilter}|${catFilter}|${sort}`} title="Choose a paper" items={items} papers={data.papers}
          colors={colors} degree={degree} hovered={null} selected={selected} loading={pending !== null} onHover={() => {}} onSelect={select} />
        <section className="reading-detail" aria-label="Paper details">
          {selected !== null && <button className="back-to-list" onClick={() => setMobileDetail(false)}>← Back to papers</button>}
          <DetailPane data={data} selected={selected} areaById={areaById} areaFilter={areaFilter} degree={degree}
            savedIds={args?.saved_ids ?? []} onSave={id => send({ type: "save", id })}
            onSelect={select} onHover={() => {}} onArea={focus} onOpenArea={id => send({ type: "open_area", id })} />
        </section>
      </div>
    </>}
  </main>;
}
