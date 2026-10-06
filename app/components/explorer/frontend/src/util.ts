import type { Paper } from "./types";

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

export function pct(score: number): string {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

export const arxivUrl = (id: string) => `https://arxiv.org/abs/${id}`;
export const pdfUrl = (id: string) => `https://arxiv.org/pdf/${id}`;
export const doiUrl = (id: string) => `https://doi.org/10.48550/arXiv.${id}`;
