import type { Paper, Payload } from "./types";

const sharedCache = new WeakMap<Payload, Map<string, number>>();

/** Number of references two papers both cite (0 when unknown or none). */
export function sharedRefs(data: Payload, a: number, b: number): number {
  let m = sharedCache.get(data);
  if (!m) {
    m = new Map();
    for (const e of data.edges) if (e.length === 4) m.set(`${e[0]}-${e[1]}`, e[3]);
    sharedCache.set(data, m);
  }
  return m.get(a < b ? `${a}-${b}` : `${b}-${a}`) ?? 0;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function fmtDate(iso: string): string {
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  if (!y || !m || !d) return iso;
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

export function authorList(p: Paper): string[] {
  return p.au ? p.au.split(";").map((s) => s.trim()).filter(Boolean) : [];
}

function lastName(name: string): string {
  const parts = name.split(/\s+/);
  return parts[parts.length - 1] || name;
}

export function shortAuthors(p: Paper): string {
  const a = authorList(p);
  if (a.length === 0) return "Unknown authors";
  if (a.length === 1) return lastName(a[0]);
  if (a.length === 2) return `${lastName(a[0])} & ${lastName(a[1])}`;
  return `${lastName(a[0])} et al.`;
}

export function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

export const arxivUrl = (id: string) => `https://arxiv.org/abs/${id}`;
export const pdfUrl = (id: string) => `https://arxiv.org/pdf/${id}`;
