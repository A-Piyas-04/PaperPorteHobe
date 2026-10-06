import { useEffect, useMemo, useRef, useState } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import { createNodeBorderProgram } from "@sigma/node-border";
import type { Payload } from "./types";
import { shortAuthors } from "./util";

export type CameraCmd =
  | { kind: "node"; i: number; zoom: boolean; n: number }
  | { kind: "nodes"; ids: number[]; n: number }
  | { kind: "reset"; n: number };

interface Props {
  data: Payload;
  colors: Map<number, string>;
  hovered: number | null;
  selected: number | null;
  highlight: Map<number, number> | null; // paper index -> rank
  areaFilter: number | null;
  catFilter: string | null;
  camera: CameraCmd | null;
  onHover: (i: number | null) => void;
  onSelect: (i: number | null) => void;
  onArea: (id: number) => void;
}

const INK = "#161a22";
const DIM_NODE = "#e4e7ec";
const EDGE = "#e9ebef";
const EDGE_FOCUS = "#7d8696";
const NOISE = "#cfd3da";
const NOISE_EMPHASIS = "#6b7383";
const FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif";

const noop = () => undefined;

interface State {
  hovered: number | null;
  selected: number | null;
  highlight: Map<number, number> | null;
  areaFilter: number | null;
  catFilter: string | null;
  nbrs: Set<string>;
}

export default function GraphView(props: Props) {
  const { data, colors } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const labelEls = useRef<{ el: HTMLDivElement; w: number; h: number }[]>([]);
  const cbs = useRef(props);
  cbs.current = props;
  const st = useRef<State>({
    hovered: null, selected: null, highlight: null, areaFilter: null, catFilter: null, nbrs: new Set(),
  });
  const [ready, setReady] = useState(false);

  const centroids = useMemo(() => {
    const acc = new Map<number, { x: number; y: number; n: number }>();
    for (const p of data.papers) {
      if (p.c < 0) continue;
      const a = acc.get(p.c) ?? { x: 0, y: 0, n: 0 };
      a.x += p.x; a.y += p.y; a.n += 1;
      acc.set(p.c, a);
    }
    return data.areas
      .filter((a) => acc.has(a.id))
      .map((a) => {
        const s = acc.get(a.id)!;
        return { ...a, x: s.x / s.n, y: s.y / s.n };
      })
      .sort((a, b) => b.size - a.size);
  }, [data]);

  // Build the graph and renderer once per dataset.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const P = data.papers;
    const g = new Graph({ type: "undirected", multi: false });

    const degree = new Array(P.length).fill(0);
    for (const [a, b] of data.edges) { degree[a] += 1; degree[b] += 1; }
    const metric = P.map((p, i) => (data.sizeBy === "citations" ? Math.log1p(p.ci) : degree[i]));
    const lo = Math.min(...metric);
    const hi = Math.max(...metric);
    const span = hi - lo || 1;

    P.forEach((p, i) => {
      g.addNode(String(i), {
        x: p.x,
        y: p.y,
        size: 3 + 8 * Math.pow((metric[i] - lo) / span, 0.9),
        color: p.c < 0 ? NOISE : colors.get(p.c) ?? NOISE,
        label: shortAuthors(p),
        borderColor: INK,
        borderSize: 0,
      });
    });
    const ws = data.edges.map((e) => e[2]);
    const wlo = Math.min(...ws);
    const wspan = Math.max(...ws) - wlo || 1;
    for (const [a, b, w] of data.edges) {
      const s = String(a), t = String(b);
      if (s === t || g.hasEdge(s, t)) continue;
      g.addEdge(s, t, { size: 0.4 + ((w - wlo) / wspan) * 0.9, color: EDGE });
    }

    const visible = (i: number) => {
      const s = st.current;
      const p = P[i];
      return (s.areaFilter === null || p.c === s.areaFilter) && (s.catFilter === null || p.cat === s.catFilter);
    };

    const renderer = new Sigma(g, container, {
      allowInvalidContainer: true,
      defaultNodeType: "bordered",
      nodeProgramClasses: {
        bordered: createNodeBorderProgram({
          borders: [
            { size: { attribute: "borderSize", defaultValue: 0, mode: "pixels" }, color: { attribute: "borderColor" } },
            { size: { fill: true }, color: { attribute: "color" } },
          ],
          drawHover: noop,
        }),
      },
      defaultDrawNodeHover: noop,
      renderEdgeLabels: false,
      labelFont: FONT,
      labelSize: 13,
      labelWeight: "600",
      labelColor: { color: "#2b313c" },
      labelDensity: 0.6,
      labelGridCellSize: 90,
      labelRenderedSizeThreshold: 9,
      zIndex: true,
      minCameraRatio: 0.04,
      maxCameraRatio: 2.5,
      stagePadding: 40,
      nodeReducer: (node, attrs) => {
        const s = st.current;
        const i = Number(node);
        const res: Record<string, unknown> = { ...attrs };
        if (!visible(i)) {
          res.hidden = true;
          return res;
        }
        const focus = s.hovered ?? s.selected;
        const emphasize = () => {
          if (P[i].c < 0) res.color = NOISE_EMPHASIS;
        };
        if (focus !== null) {
          if (i === focus) {
            res.zIndex = 3;
            res.forceLabel = true;
            res.size = (attrs.size as number) + 2;
            emphasize();
          } else if (s.nbrs.has(node)) {
            res.zIndex = 2;
            res.forceLabel = true;
            emphasize();
          } else {
            res.color = DIM_NODE;
            res.label = "";
            res.zIndex = 0;
          }
        } else if (s.highlight) {
          const rank = s.highlight.get(i);
          if (rank !== undefined) {
            res.zIndex = 2;
            res.size = (attrs.size as number) + 1.5;
            emphasize();
            if (rank < 10) res.forceLabel = true;
          } else {
            res.color = DIM_NODE;
            res.label = "";
            res.zIndex = 0;
          }
        }
        if (i === s.selected) {
          res.borderSize = 3;
          res.zIndex = 4;
          res.forceLabel = true;
        } else if (i === s.hovered) {
          res.borderSize = 2;
        }
        return res;
      },
      edgeReducer: (edge, attrs) => {
        const s = st.current;
        const [a, b] = g.extremities(edge);
        const ia = Number(a), ib = Number(b);
        const res: Record<string, unknown> = { ...attrs };
        if (!visible(ia) || !visible(ib)) {
          res.hidden = true;
          return res;
        }
        const focus = s.hovered ?? s.selected;
        if (focus !== null) {
          if (ia === focus || ib === focus) {
            res.color = EDGE_FOCUS;
            res.size = (attrs.size as number) + 0.8;
            res.zIndex = 1;
          } else {
            res.hidden = true;
          }
        } else if (s.highlight) {
          if (!(s.highlight.has(ia) && s.highlight.has(ib))) res.hidden = true;
          else res.color = "#c4cad4";
        }
        return res;
      },
    });

    renderer.on("enterNode", ({ node }) => {
      container.style.cursor = "pointer";
      cbs.current.onHover(Number(node));
    });
    renderer.on("leaveNode", () => {
      container.style.cursor = "";
      cbs.current.onHover(null);
    });
    renderer.on("clickNode", ({ node }) => cbs.current.onSelect(Number(node)));
    renderer.on("clickStage", () => cbs.current.onSelect(null));

    // Area labels, drawn as HTML so they stay crisp and clickable.
    const overlay = overlayRef.current!;
    overlay.innerHTML = "";
    labelEls.current = centroids.map((a) => {
      const el = document.createElement("div");
      el.className = "area-label";
      el.innerHTML = `<span class="dot" style="background:${a.color}"></span>`;
      const txt = document.createElement("span");
      txt.textContent = a.name;
      el.appendChild(txt);
      el.onclick = () => cbs.current.onArea(a.id);
      overlay.appendChild(el);
      return { el, w: el.offsetWidth, h: el.offsetHeight };
    });

    const drawOverlay = () => {
      const s = st.current;
      const ratio = renderer.getCamera().ratio;
      const focusActive = s.hovered !== null || s.selected !== null || s.highlight !== null;
      const limit = ratio > 0.8 ? 10 : ratio > 0.4 ? 18 : 999;
      const placed: { x: number; y: number; w: number; h: number }[] = [];
      const { width, height } = renderer.getDimensions();
      centroids.forEach((a, k) => {
        const item = labelEls.current[k];
        if (!item) return;
        const { el } = item;
        if (!item.w) { item.w = el.offsetWidth; item.h = el.offsetHeight; }
        let show = k < limit && (s.areaFilter === null || s.areaFilter === a.id);
        const pos = renderer.graphToViewport({ x: a.x, y: a.y });
        if (show && (pos.x < 0 || pos.y < 0 || pos.x > width || pos.y > height)) show = false;
        const box = {
          x: Math.max(6, Math.min(width - item.w - 6, pos.x - item.w / 2)),
          y: Math.max(6, Math.min(height - item.h - 6, pos.y - item.h / 2)),
          w: item.w,
          h: item.h,
        };
        if (show && placed.some((b) => box.x < b.x + b.w + 6 && box.x + box.w + 6 > b.x && box.y < b.y + b.h + 4 && box.y + box.h + 4 > b.y)) show = false;
        if (show) placed.push(box);
        el.style.transform = `translate(${box.x}px, ${box.y}px)`;
        el.style.opacity = show ? (focusActive ? "0.28" : "1") : "0";
        el.style.pointerEvents = show && !focusActive ? "auto" : "none";
      });

      const tip = tipRef.current;
      if (tip) {
        const h = s.hovered;
        if (h !== null && g.hasNode(String(h)) && visible(h)) {
          const at = g.getNodeAttributes(String(h));
          const pos = renderer.graphToViewport({ x: at.x as number, y: at.y as number });
          tip.style.transform = `translate(${pos.x}px, ${pos.y - 16}px) translate(-50%, -100%)`;
          tip.style.opacity = "1";
        } else {
          tip.style.opacity = "0";
        }
      }
    };
    renderer.on("afterRender", drawOverlay);

    sigmaRef.current = renderer;
    graphRef.current = g;
    const raf = requestAnimationFrame(() => setReady(true));
    return () => {
      cancelAnimationFrame(raf);
      renderer.kill();
      sigmaRef.current = null;
      graphRef.current = null;
      setReady(false);
    };
  }, [data, colors, centroids]);

  // Push interaction state into the reducers.
  useEffect(() => {
    const g = graphRef.current;
    const r = sigmaRef.current;
    if (!g || !r) return;
    const focus = props.hovered ?? props.selected;
    st.current = {
      hovered: props.hovered,
      selected: props.selected,
      highlight: props.highlight,
      areaFilter: props.areaFilter,
      catFilter: props.catFilter,
      nbrs: focus !== null && g.hasNode(String(focus)) ? new Set(g.neighbors(String(focus))) : new Set(),
    };
    const tip = tipRef.current;
    if (tip) {
      const p = props.hovered !== null ? data.papers[props.hovered] : null;
      tip.innerHTML = "";
      if (p) {
        const t = document.createElement("div");
        t.className = "t";
        t.textContent = p.t;
        const m = document.createElement("div");
        m.className = "m";
        m.textContent = shortAuthors(p);
        tip.append(t, m);
      }
    }
    r.refresh();
  }, [props.hovered, props.selected, props.highlight, props.areaFilter, props.catFilter, data]);

  // Camera moves.
  useEffect(() => {
    const r = sigmaRef.current;
    const cmd = props.camera;
    if (!r || !cmd) return;
    const cam = r.getCamera();
    if (cmd.kind === "reset") {
      cam.animatedReset({ duration: 650 });
      return;
    }
    if (cmd.kind === "node") {
      const d = r.getNodeDisplayData(String(cmd.i));
      if (!d) return;
      const ratio = cmd.zoom ? Math.min(cam.ratio, 0.35) : cam.ratio;
      cam.animate({ x: d.x, y: d.y, ratio }, { duration: 650, easing: "cubicInOut" });
      return;
    }
    const pts = cmd.ids.map((i) => r.getNodeDisplayData(String(i))).filter(Boolean) as { x: number; y: number }[];
    if (!pts.length) return;
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const ratio = Math.min(1.1, Math.max(0.12, extent * 1.25));
    cam.animate({ x: cx, y: cy, ratio }, { duration: 700, easing: "cubicInOut" });
  }, [props.camera]);

  return (
    <div className={`graph-wrap${ready ? " ready" : ""}`}>
      <div ref={containerRef} className="graph-canvas" />
      <div ref={overlayRef} className="graph-overlay" />
      <div ref={tipRef} className="graph-tip" />
      {!ready && <div className="graph-loading"><div className="spinner" /></div>}
    </div>
  );
}
