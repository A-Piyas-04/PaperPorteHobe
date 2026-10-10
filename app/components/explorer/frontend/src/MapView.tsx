import { useEffect, useMemo, useRef, useState } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import { createNodeBorderProgram } from "@sigma/node-border";
import type { Area, Paper, Payload, Scope } from "./types";
import { fmtInt, shortAuthors } from "./util";

export interface CameraCmd {
  kind: "zoomIn" | "zoomOut" | "fit";
  n: number;
}

interface Props {
  data: Payload;
  colors: Map<number, string>;
  scope: Scope;
  searchNonce: number; // identifies a particular search result set
  searchItems: number[]; // paper indices for scope "search"
  selected: number | null;
  focusNonce: number; // bump to re-centre the camera on `selected`
  cameraCmd: CameraCmd | null;
  reducedMotion: boolean;
  onSelectPaper: (i: number) => void;
  onFocusArea: (id: number) => void;
  onClearSelection: () => void;
}

const INK = "#161a22";
const DIM_NODE = "#dfe3ea";
const DIM_LABEL = "#b7bdc9";
const EDGE_FAINT = "#e7e9ee";
const EDGE_FOCUS = "#8a93a3";
const NOISE = "#cbd0d8";
const FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif";
const noop = () => undefined;

type Scene =
  | { kind: "areas"; key: string; areas: Area[]; edges: [number, number, number][] }
  | { kind: "papers"; key: string; ids: number[]; edges: [number, number, number][]; root: number | null };

/** Deterministic description of what the map should draw for a given scope. */
function buildScene(data: Payload, scope: Scope, searchNonce: number, searchItems: number[]): Scene {
  if (scope.kind === "overview") {
    const areas = data.areas.filter((a) => (a.rc ?? 0) > 0 && a.x != null && a.y != null);
    const present = new Set(areas.map((a) => a.id));
    const edges = data.areaEdges.filter(([a, b]) => present.has(a) && present.has(b));
    return { kind: "areas", key: "overview", areas, edges };
  }
  if (scope.kind === "area") {
    const ids = data.papers
      .filter((p) => p.c === scope.areaId)
      .sort((a, b) => b.ci - a.ci || b.d.localeCompare(a.d))
      .slice(0, data.maxAreaPapers)
      .map((p) => p.i);
    return papersScene(`area:${scope.areaId}`, data, ids, null);
  }
  if (scope.kind === "search") {
    const ids = searchItems.slice(0, data.maxAreaPapers);
    return papersScene(`search:${searchNonce}`, data, ids, null);
  }
  // paper neighbourhood, breadth-first up to `depth`
  const seen = new Set<number>([scope.root]);
  let frontier = [scope.root];
  for (let d = 0; d < scope.depth && seen.size < data.maxAreaPapers; d++) {
    const next: number[] = [];
    for (const i of frontier) {
      for (const [j] of data.neighbors[String(i)] ?? []) {
        if (!seen.has(j) && seen.size < data.maxAreaPapers) {
          seen.add(j);
          next.push(j);
        }
      }
    }
    frontier = next;
  }
  return papersScene(`paper:${scope.root}:${scope.depth}`, data, [...seen], scope.root);
}

function papersScene(key: string, data: Payload, ids: number[], root: number | null): Scene {
  const set = new Set(ids);
  const edges: [number, number, number][] = [];
  const seenPair = new Set<string>();
  const push = (a: number, b: number, w: number) => {
    if (a === b || !set.has(a) || !set.has(b)) return;
    const pk = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (seenPair.has(pk)) return;
    seenPair.add(pk);
    edges.push([a, b, w]);
  };
  for (const [a, b, w] of data.edges) push(a, b, w);
  // Guarantee a selected paper's own links show even if the global edge list
  // (symmetric kNN) happens not to include every direction.
  if (root != null) for (const [j, s] of data.neighbors[String(root)] ?? []) push(root, j, s);
  return { kind: "papers", key, ids, edges, root };
}

export default function MapView(props: Props) {
  const { data, colors } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const cbs = useRef(props);
  cbs.current = props;
  const [failed, setFailed] = useState(false);

  const scene = useMemo(
    () => buildScene(data, props.scope, props.searchNonce, props.searchItems),
    [data, props.scope, props.searchNonce, props.searchItems],
  );

  // Mutable interaction state the reducers read, so hover/selection never
  // force a full React re-render of the renderer.
  const st = useRef<{ selected: number | null; neighbors: Set<string>; hovered: string | null; scene: Scene }>({
    selected: null,
    neighbors: new Set(),
    hovered: null,
    scene,
  });
  st.current.scene = scene;

  // ---- create the renderer once, for the life of the component ----------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const g = new Graph({ type: "undirected", multi: false });
    let renderer: Sigma;
    try {
      renderer = new Sigma(g, container, {
        allowInvalidContainer: true,
        defaultNodeType: "bordered",
        nodeProgramClasses: {
          bordered: createNodeBorderProgram({
            borders: [
              { size: { attribute: "ring", defaultValue: 0, mode: "pixels" }, color: { attribute: "ringColor" } },
              { size: { fill: true }, color: { attribute: "color" } },
            ],
            drawHover: noop,
          }),
        },
        defaultDrawNodeHover: noop,
        renderEdgeLabels: false,
        labelFont: FONT,
        labelSize: 12,
        labelWeight: "600",
        labelColor: { color: "#2b313c" },
        labelDensity: 1,
        labelGridCellSize: 75,
        labelRenderedSizeThreshold: 6,
        zIndex: true,
        minCameraRatio: 0.05,
        maxCameraRatio: 3,
        stagePadding: 60,
        nodeReducer: (node, attrs) => {
          const s = st.current;
          const res: Record<string, unknown> = { ...attrs };
          // Areas keep Sigma's collision-aware labelling (no forced labels),
          // so the overview stays readable; hover shows any hidden name.
          if (s.scene.kind === "areas") return res;
          if (s.selected === null) return res;
          const sel = String(s.selected);
          if (node === sel) {
            res.ring = 3;
            res.ringColor = INK;
            res.size = (attrs.size as number) + 2;
            res.forceLabel = true;
            res.zIndex = 3;
          } else if (s.neighbors.has(node)) {
            res.forceLabel = true;
            res.zIndex = 2;
          } else {
            res.color = DIM_NODE;
            res.labelColor = DIM_LABEL;
            res.label = "";
            res.zIndex = 0;
          }
          return res;
        },
        edgeReducer: (edge, attrs) => {
          const s = st.current;
          const res: Record<string, unknown> = { ...attrs };
          if (s.scene.kind === "areas") return res;
          if (s.selected === null) return res;
          const [a, b] = g.extremities(edge);
          const sel = String(s.selected);
          if (a === sel || b === sel) {
            res.color = EDGE_FOCUS;
            res.size = (attrs.size as number) + 0.8;
            res.zIndex = 1;
          } else {
            res.hidden = true;
          }
          return res;
        },
      });
    } catch {
      setFailed(true);
      return;
    }

    renderer.on("enterNode", ({ node }) => {
      container.style.cursor = "pointer";
      st.current.hovered = node;
      positionTip(renderer, node);
    });
    renderer.on("leaveNode", () => {
      container.style.cursor = "";
      st.current.hovered = null;
      if (tipRef.current) tipRef.current.style.opacity = "0";
    });
    renderer.on("clickNode", ({ node }) => {
      const c = cbs.current;
      if (st.current.scene.kind === "areas") c.onFocusArea(Number(node.slice(1)));
      else c.onSelectPaper(Number(node));
    });
    renderer.on("clickStage", () => {
      if (st.current.scene.kind !== "areas") cbs.current.onClearSelection();
    });
    renderer.on("afterRender", () => {
      const h = st.current.hovered;
      if (h) positionTip(renderer, h);
    });

    const positionTip = (r: Sigma, node: string) => {
      const tip = tipRef.current;
      if (!tip || !r.getGraph().hasNode(node)) return;
      const d = r.getNodeDisplayData(node);
      if (!d) return;
      const p = r.graphToViewport({ x: d.x, y: d.y });
      tip.innerHTML = tipHtml(node);
      tip.style.transform = `translate(${p.x}px, ${p.y - 18}px) translate(-50%, -100%)`;
      tip.style.opacity = "1";
    };

    const tipHtml = (node: string): string => {
      const esc = (s: string) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c] || c);
      if (node.startsWith("a")) {
        const a = data.areas.find((x) => x.id === Number(node.slice(1)));
        if (!a) return "";
        return `<div class="t">${esc(a.name)}</div><div class="m">${fmtInt(a.size)} papers · click to open</div>`;
      }
      const p = data.papers[Number(node)];
      if (!p) return "";
      return `<div class="t">${esc(p.t)}</div><div class="m">${esc(shortAuthors(p))}</div>`;
    };

    sigmaRef.current = renderer;
    graphRef.current = g;
    return () => {
      renderer.kill();
      sigmaRef.current = null;
      graphRef.current = null;
    };
    // The renderer is created exactly once; data is stable per bundle version.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.version]);

  // ---- (re)populate the graph when the scene changes --------------------
  useEffect(() => {
    const g = graphRef.current;
    const r = sigmaRef.current;
    if (!g || !r) return;
    g.clear();

    if (scene.kind === "areas") {
      const sizes = scene.areas.map((a) => a.rc ?? 1);
      const lo = Math.min(...sizes, 1);
      const hi = Math.max(...sizes, 1);
      const span = hi - lo || 1;
      for (const a of scene.areas) {
        g.addNode("a" + a.id, {
          x: a.x!,
          y: a.y!,
          size: 10 + 20 * Math.sqrt(((a.rc ?? 1) - lo) / span),
          color: a.color,
          label: a.name,
          ring: 0,
          ringColor: INK,
        });
      }
      const ws = scene.edges.map((e) => e[2]);
      const wlo = Math.min(...ws, 0);
      const wspan = Math.max(...ws, 1) - wlo || 1;
      for (const [a, b, w] of scene.edges) {
        const sa = "a" + a, sb = "a" + b;
        if (!g.hasNode(sa) || !g.hasNode(sb) || g.hasEdge(sa, sb)) continue;
        g.addEdge(sa, sb, { size: 0.6 + ((w - wlo) / wspan) * 3.2, color: "#d4d8e0", zIndex: 0 });
      }
    } else {
      const ids = scene.ids;
      const degree = new Map<number, number>();
      for (const [a, b] of scene.edges) {
        degree.set(a, (degree.get(a) ?? 0) + 1);
        degree.set(b, (degree.get(b) ?? 0) + 1);
      }
      const metric = (i: number) =>
        data.sizeBy === "citations" ? Math.log1p(data.papers[i].ci) : degree.get(i) ?? 0;
      const vals = ids.map(metric);
      const lo = Math.min(...vals, 0);
      const span = (Math.max(...vals, 1) - lo) || 1;
      for (const i of ids) {
        const p: Paper = data.papers[i];
        g.addNode(String(i), {
          x: p.x,
          y: p.y,
          size: 4 + 9 * Math.pow((metric(i) - lo) / span, 0.9),
          color: p.c < 0 ? NOISE : colors.get(p.c) ?? NOISE,
          label: p.t,
          ring: 0,
          ringColor: INK,
        });
      }
      const ws = scene.edges.map((e) => e[2]);
      const wlo = Math.min(...ws, 0);
      const wspan = (Math.max(...ws, 1) - wlo) || 1;
      for (const [a, b, w] of scene.edges) {
        const sa = String(a), sb = String(b);
        if (!g.hasNode(sa) || !g.hasNode(sb) || g.hasEdge(sa, sb)) continue;
        g.addEdge(sa, sb, { size: 0.5 + ((w - wlo) / wspan) * 1.4, color: EDGE_FAINT, zIndex: 0 });
      }
    }

    // Recompute the selected paper's neighbour set for the new scene.
    refreshSelection();
    r.refresh();
    const cam = r.getCamera();
    cam.animatedReset({ duration: props.reducedMotion ? 0 : 600 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene.key]);

  // ---- selection emphasis (no rebuild, keeps camera/viewport) -----------
  const refreshSelection = () => {
    const g = graphRef.current;
    const s = st.current;
    s.selected = props.selected;
    s.neighbors = new Set();
    if (props.selected !== null && g && g.hasNode(String(props.selected))) {
      for (const n of g.neighbors(String(props.selected))) s.neighbors.add(n);
    }
  };
  useEffect(() => {
    const r = sigmaRef.current;
    if (!r) return;
    refreshSelection();
    r.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.selected]);

  // ---- deliberate camera moves only -------------------------------------
  useEffect(() => {
    const r = sigmaRef.current;
    const cmd = props.cameraCmd;
    if (!r || !cmd) return;
    const cam = r.getCamera();
    const dur = props.reducedMotion ? 0 : 400;
    if (cmd.kind === "zoomIn") cam.animatedZoom({ duration: dur });
    else if (cmd.kind === "zoomOut") cam.animatedUnzoom({ duration: dur });
    else cam.animatedReset({ duration: props.reducedMotion ? 0 : 600 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.cameraCmd]);

  // Centre on the selected paper only when selection arrives from outside the
  // map (a related-paper jump or the list), never on a direct map click/hover.
  useEffect(() => {
    const r = sigmaRef.current;
    if (!r || props.selected === null || props.focusNonce === 0) return;
    const d = r.getNodeDisplayData(String(props.selected));
    if (!d) return;
    const cam = r.getCamera();
    cam.animate({ x: d.x, y: d.y, ratio: Math.min(cam.ratio, 0.5) },
      { duration: props.reducedMotion ? 0 : 500 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.focusNonce]);

  if (failed) {
    return (
      <div className="map-fallback" role="status">
        <b>Interactive map unavailable</b>
        <p>Your browser could not start the graphics renderer this map needs. Use the List
          view to search, read papers and follow related work — every paper stays reachable there.</p>
      </div>
    );
  }

  return (
    <div className="map-canvas-wrap">
      <div
        ref={containerRef}
        className="map-canvas"
        role="application"
        aria-label="Research map. A visual graph of research areas and papers. Use the List view for a keyboard-accessible paper list."
      />
      <div ref={tipRef} className="map-tip" aria-hidden />
    </div>
  );
}
