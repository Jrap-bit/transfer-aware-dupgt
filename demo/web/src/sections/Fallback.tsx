import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { Recommendation } from "../lib/api";
import { cn } from "../lib/utils";
import { cleanTitle } from "../components/ui/poster";
import { AnimatedNumber } from "../components/ui/number";

const x = (v: number, lo: number, hi: number) => `${((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * 100}%`;

// Step: open up the uncertainty-aware fallback for one movie (Method A, with Method B alongside).
export function FallbackExplainer({ rec, initial }: { rec: Recommendation; initial?: string }) {
  const rows = rec.fallback.filter(f => f.bridges);
  const [i, setI] = useState(() => Math.max(0, rows.findIndex(r => r.title === initial)));
  if (!rows.length || !rec.blend_params) return <div className="text-sm text-neutral-500">The fallback models are still loading for this setting.</div>;
  const f = rows[Math.min(i, rows.length - 1)];
  const p = rec.blend_params;
  const values = [...f.bridges!, f.movie_avg, f.final!, f.probabilistic?.mu ?? f.mu!];
  const lo = Math.max(1, Math.floor((Math.min(...values) - 0.15) * 4) / 4), hi = Math.min(5.2, Math.ceil((Math.max(...values) + 0.15) * 4) / 4);
  const sigma2 = p.tau2 + p.c * f.d! ** 2;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
      <div className="rounded-2xl border border-white/[0.07] bg-gradient-to-b from-white/[0.03] to-transparent p-6">
        <div className="mb-5 flex flex-wrap gap-1.5">
          {rows.map((r, k) => (
            <button key={r.title} onClick={() => setI(k)}
              className={cn("rounded-full border px-2.5 py-1 text-xs transition", k === i ? "border-yellow-300/60 bg-yellow-300/10 text-yellow-100" : "border-white/10 text-neutral-400 hover:text-neutral-200")}>
              {cleanTitle(r.title)}
            </button>
          ))}
        </div>

        {/* number line */}
        <div className="relative mx-10 mt-10 h-48">
          <div className="absolute inset-x-0 top-20 h-px bg-white/15" />
          {Array.from({ length: Math.round((hi - lo) / 0.25) + 1 }, (_, k) => lo + k * 0.25).map(t => (
            <div key={t} className="absolute top-20" style={{ left: x(t, lo, hi) }}>
              <span className="absolute h-2 w-px -translate-y-1/2 bg-white/20" />
              {Math.abs(t - Math.round(t * 2) / 2) < 1e-6 && <span className="absolute top-3 -translate-x-1/2 font-mono text-[10px] text-neutral-500">{t.toFixed(1)}★</span>}
            </div>
          ))}
          {/* 5 bridges */}
          {f.bridges!.map((b, k) => (
            <motion.div key={k} className="absolute top-[52px] -translate-x-1/2" animate={{ left: x(b, lo, hi) }} transition={{ type: "spring", bounce: 0.25 }}>
              <span className="block h-3.5 w-3.5 rounded-full bg-cyan-400 shadow-[0_0_14px_rgba(34,211,238,.7)] ring-2 ring-neutral-950" />
            </motion.div>
          ))}
          <Marker label="ensemble mean μ" v={f.mu!} lo={lo} hi={hi} color="#22d3ee" top={4} />
          <Marker label="movie average m" v={f.movie_avg} lo={lo} hi={hi} color="#a3a3a3" top={98} below />
          <Marker label="fallback r̂" v={f.final!} lo={lo} hi={hi} color="#facc15" top={98} below big stem={44} />
          {/* disagreement band */}
          <motion.div className="absolute top-[47px] h-6 rounded-full bg-cyan-400/10 ring-1 ring-cyan-400/30"
            animate={{ left: x(f.mu! - f.d!, lo, hi), width: `calc(${x(f.mu! + f.d!, lo, hi)} - ${x(f.mu! - f.d!, lo, hi)})` }} />
        </div>
        <p className="mt-2 text-sm text-neutral-400">
          Five bridges trained from different random starts (<span className="text-cyan-300">cyan</span>). Their spread is the
          disagreement <span className="font-mono text-neutral-200">d</span>. The wider the spread, the more the final rating is pulled from
          <span className="text-cyan-300"> μ</span> towards the <span className="text-neutral-200">movie average</span>.
        </p>
      </div>

      <div className="space-y-4">
        <div className="rounded-2xl border border-yellow-300/20 bg-yellow-300/[0.04] p-6">
          <p className="text-xs font-medium uppercase tracking-wider text-yellow-200/70">Method A · uncertainty blend</p>
          <div className="mt-4 space-y-2.5 font-mono text-[13px] text-neutral-300">
            <Row k="disagreement d" v={<AnimatedNumber value={f.d!} digits={3} />} />
            <Row k="σ² = τ² + c·d²" v={<><AnimatedNumber value={p.tau2} digits={2} /> + {p.c.toFixed(1)}·d² = <AnimatedNumber value={sigma2} digits={3} /></>} />
            <Row k="w = s₀ / (s₀ + σ²)" v={<span className="text-yellow-200"><AnimatedNumber value={f.w!} digits={3} /></span>} />
            <div className="my-3 h-px bg-white/10" />
            <Row k="r̂ = w·μ + (1−w)·m" v={<span className="text-lg font-semibold text-yellow-300"><AnimatedNumber value={f.final!} digits={2} />★</span>} />
          </div>
          <TrustBar w={f.w!} />
        </div>
        <div className="grid gap-3">
          {f.probabilistic && <Mini title="Method B · probabilistic" sub={`own σ² → w = ${f.probabilistic.w.toFixed(2)}`} value={f.probabilistic.final} color="#34d399" />}
        </div>
      </div>
    </div>
  );
}

function Marker({ label, v, lo, hi, color, top, below, big, stem = 16 }: { label: string; v: number; lo: number; hi: number; color: string; top: number; below?: boolean; big?: boolean; stem?: number }) {
  return (
    <motion.div className="absolute -translate-x-1/2" style={{ top }} animate={{ left: x(v, lo, hi) }} transition={{ type: "spring", bounce: 0.25 }}>
      <div className="flex flex-col items-center gap-1">
        {below && <span className="w-px" style={{ background: color, height: stem }} />}
        <span className={cn("whitespace-nowrap rounded-full px-2 py-0.5 font-mono text-[11px]", big && "text-xs font-semibold")}
          style={{ color: big ? "#0a0a0a" : color, background: big ? color : `${color}1f`, border: `1px solid ${color}55` }}>
          {label} {v.toFixed(2)}
        </span>
        {!below && <span className="h-5 w-px" style={{ background: color }} />}
      </div>
    </motion.div>
  );
}

const Row = ({ k, v }: { k: string; v: React.ReactNode }) => (
  <div className="flex items-baseline justify-between gap-3"><span className="text-neutral-500">{k}</span><span>{v}</span></div>
);

function TrustBar({ w }: { w: number }) {
  return (
    <div className="mt-5">
      <div className="flex justify-between text-[11px] text-neutral-500"><span>trust the movie average</span><span>trust the bridges</span></div>
      <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-white/10">
        <motion.div className="h-full rounded-full bg-gradient-to-r from-yellow-500 to-yellow-300" animate={{ width: `${w * 100}%` }} transition={{ type: "spring", bounce: 0.15 }} />
      </div>
    </div>
  );
}

function Mini({ title, sub, value, color }: { title: string; sub: string; value: number; color: string }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div className="rounded-2xl bg-white/[0.03] p-4 ring-1 ring-inset ring-white/[0.06]">
        <p className="text-xs text-neutral-400">{title}</p>
        <p className="mt-1 font-mono text-xl font-semibold" style={{ color }}><AnimatedNumber value={value} />★</p>
        <p className="mt-1 text-[11px] text-neutral-500">{sub}</p>
      </motion.div>
    </AnimatePresence>
  );
}
