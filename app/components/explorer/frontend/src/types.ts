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
  size: number;
}

export interface Payload {
  version: string;
  papers: Paper[];
  edges: [number, number, number][];
  neighbors: Record<string, [number, number][]>;
  areas: Area[];
  categories: [string, number][];
  sizeBy: "citations" | "links";
}

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
  select: string | null;
  height: number;
}

export type UIEvent =
  | { type: "search"; q: string; nonce: number }
  | { type: "clear"; nonce: number }
  | { type: "open_area"; id: number; nonce: number };
