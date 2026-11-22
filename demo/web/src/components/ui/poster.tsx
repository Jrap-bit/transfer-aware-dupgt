import { cn } from "../../lib/utils";

const PALETTES = [
  ["#1e1b4b", "#7c3aed"], ["#172554", "#0ea5e9"], ["#3b0764", "#db2777"], ["#052e16", "#10b981"],
  ["#431407", "#f59e0b"], ["#1c1917", "#ef4444"], ["#082f49", "#22d3ee"], ["#2e1065", "#f472b6"],
];
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
export const cleanTitle = (t: string) => t.replace(/\s*[\[(].*?[\])]/g, "").replace(/\s+-\s+.*$/, "").trim() || t;

// A generated "poster": gradient plus title typography (no external images needed offline).
export function Poster({ title, className }: { title: string; className?: string }) {
  const [a, b] = PALETTES[hash(title) % PALETTES.length];
  const clean = cleanTitle(title);
  return (
    <div className={cn("relative flex aspect-[4/5] w-full flex-col justify-end overflow-hidden rounded-xl p-3", className)}
      style={{ background: `radial-gradient(120% 80% at 20% 0%, ${b}cc, transparent 60%), linear-gradient(160deg, ${a}, #09090b 85%)` }}>
      <div className="absolute inset-0 [background-image:radial-gradient(rgba(255,255,255,.12)_1px,transparent_1px)] [background-size:10px_10px] opacity-40 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
      <div className="relative line-clamp-3 text-base font-semibold leading-tight tracking-tight text-white drop-shadow">{clean}</div>
    </div>
  );
}
