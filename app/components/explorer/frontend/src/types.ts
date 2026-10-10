export interface Paper {
  i: number;
  id: string;
  t: string; // title
  au: string; // "A; B; C"
  d: string; // ISO date
  c: number; // area id (-1 = unassigned)
  cat: string;
  x: number;
  y: number;
  ci: number; // cited by
  rf: number; // references
  v: string; // venue
  ab: string; // abstract (truncated)
}

export interface Area {
  id: number;
  name: string;
  color: string;
  size: number; // papers in the corpus for this area
  rc?: number; // papers from this area rendered in the current sample
  x?: number; // embedding centroid (only when rc > 0)
  y?: number;
}

export interface Payload {
  version: string;
  papers: Paper[];
  edges: [number, number, number][];
  neighbors: Record<string, [number, number][]>;
  areas: Area[];
  areaEdges: [number, number, number][]; // [areaId, areaId, summed weight]
  categories: [string, number][];
  sizeBy: "citations" | "links";
  corpusTotal: number;
  renderedTotal: number;
  maxAreaPapers: number;
}

// Which nodes the map is currently drawing.
export type Scope =
  | { kind: "overview" }
  | { kind: "area"; areaId: number }
  | { kind: "search" }
  | { kind: "paper"; root: number; depth: number };

export interface Highlight {
  query: string;
  items: [number, number][]; // [paper index, score]
  req?: number | null;
  ambiguous?: boolean;
}

export interface Args {
  data: Payload | null;
  highlight: Highlight | null;
  focus_area: number | null;
  select: number | null;
  height: number;
  saved_ids?: string[];
}

export type UIEvent =
  | { type: "search"; q: string; nonce: number }
  | { type: "clear"; nonce: number }
  | { type: "open_area"; id: number; nonce: number }
  | { type: "save"; id: string; nonce: number };
