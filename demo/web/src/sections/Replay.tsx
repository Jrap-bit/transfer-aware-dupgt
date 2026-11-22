import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, ArrowRight, Loader2, Trophy } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, MODEL_COLORS, SHORT, type Replay as ReplayData, type ReplayUser } from "../lib/api";
import { cn } from "../lib/utils";
import { Tabs } from "../components/ui/tabs";
import { StarRow } from "../components/ui/stars";
import { cleanTitle } from "../components/ui/poster";
import { OverlapToggle } from "../components/ui/overlap-toggle";

type Bucket = "all" | "helped" | "neutral" | "harmed";
const BUCKET_TEXT: Record<string, { label: string; tone: string }> = {
  helped: { label: "DUPGT beat the movie average for this person", tone: "text-emerald-300 bg-emerald-400/10 border-emerald-400/20" },
  neutral: { label: "DUPGT and the movie average were about equal", tone: "text-neutral-300 bg-white/5 border-white/10" },
  harmed: { label: "Negative transfer: DUPGT did worse than the movie average", tone: "text-rose-300 bg-rose-400/10 border-rose-400/20" },
};
const STORY: Record<string, { title: string; body: string }> = {
  "1%": {
    title: "Only 890 shared users: transfer hurts, the fallback rescues it",
    body: "Every model without a fallback is worse than simply using the movie average. Blending towards the average turns that loss into a clear gain, on every seed (p < 0.001).",
  },
  "10%": {
    title: "8,903 shared users: bridges start to help",
    body: "The semantic gate beats DUPGT, and the uncertainty blend already beats DUPGT trained with all 89,031 shared users (RMSE 1.0119 on the 100% setting).",
  },
  "100%": {
    title: "All 89,031 shared users: the published setting",
    body: "DUPGT clearly beats the movie average and the semantic gate adds a small, consistent gain. The LLM teacher is no better than the shuffled control. Few real platforms share this many users.",
  },
};
const DEFAULT_HIDDEN = ["+ Shuffled teacher"];
const LEFT_OUT = new Set(["Ensemble mean", "Constant blend"]);   // ablations of the uncertainty blend; not shown in the demo

export function Replay() {
  const [data, setData] = useState<ReplayData | null>(null);
  const [overlap, setOverlap] = useState("10%");
  const [bucket, setBucket] = useState<Bucket>("all");
  const [userId, setUserId] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set(DEFAULT_HIDDEN));

  useEffect(() => { api.replay().then(setData); }, []);
  const bucketOf = (u: ReplayUser) => {
    const m = u.by_overlap[overlap].mae, d = m["DUPGT"] - m["Movie average"];
    return d < -0.3 ? "helped" : d > 0.3 ? "harmed" : "neutral";
  };
  const pool = useMemo(() => data ? data.users.filter(u => bucket === "all" || bucketOf(u) === bucket) : [], [data, bucket, overlap]);
  if (!data) return <div className="grid h-64 place-items-center"><Loader2 className="h-5 w-5 animate-spin text-neutral-500" /></div>;
  const idx = Math.max(0, pool.findIndex(u => u.user === userId));
  const user = pool[idx] ?? data.users[0];
  const go = (step: number) => pool.length && setUserId(pool[(idx + step + pool.length) % pool.length].user);
  const methods = data.overlaps[overlap].filter(m => !LEFT_OUT.has(m));
  const shown = methods.filter(m => !hidden.has(m));
  const toggle = (m: string) => setHidden(h => { const n = new Set(h); n.has(m) ? n.delete(m) : n.add(m); return n; });
  const view = user.by_overlap[overlap];

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <OverlapToggle id="replay-overlap" value={overlap} onChange={setOverlap} options={["1%", "10%", "100%"]} />
        <div className="flex items-center gap-2">
          <Tabs id="bucket" size="sm" value={bucket} onChange={v => { setBucket(v); setUserId(null); }}
            options={[{ value: "all", label: "All" }, { value: "helped", label: "DUPGT helped" }, { value: "neutral", label: "Same" }, { value: "harmed", label: "DUPGT hurt" }]} />
          <button onClick={() => go(-1)} className="grid h-9 w-9 place-items-center rounded-full border border-white/10 hover:bg-white/5"><ArrowLeft className="h-4 w-4" /></button>
          <span className="w-16 text-center font-mono text-xs text-neutral-500">{idx + 1}/{pool.length}</span>
          <button onClick={() => go(1)} className="grid h-9 w-9 place-items-center rounded-full bg-white text-neutral-950 hover:bg-neutral-200"><ArrowRight className="h-4 w-4" /></button>
        </div>
      </div>

      <Headline overlap={overlap} data={data} />

      <AnimatePresence mode="wait">
        <motion.div key={user.user + overlap} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}
          className="grid gap-6 xl:grid-cols-[1fr_330px]">
          <div className="relative rounded-2xl border border-white/[0.07] bg-neutral-950/60 p-6">
            <div className="absolute inset-x-12 -top-px h-px bg-gradient-to-r from-transparent via-violet-400/70 to-transparent" />
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <span className="font-mono text-sm text-neutral-300">user …{user.user}</span>
              <span className="text-xs text-neutral-500">{user.n_books} book ratings · {user.movies.length} hidden movie ratings · never seen in training</span>
              <span className={cn("rounded-full border px-2.5 py-0.5 text-xs", BUCKET_TEXT[bucketOf(user)].tone)}>{BUCKET_TEXT[bucketOf(user)].label}</span>
            </div>
            <div className="mb-6 flex flex-wrap gap-1.5">
              {user.books.slice(0, 16).map((b, k) => (
                <span key={k} className={cn("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs",
                  b.stars >= 4 ? "border-amber-400/20 bg-amber-400/[0.06] text-amber-100" : "border-white/10 bg-white/[0.03] text-neutral-400")}>
                  <span className="font-mono text-[10px]">{b.stars}★</span>{cleanTitle(b.title).slice(0, 40)}
                </span>
              ))}
            </div>
            <Legend methods={methods} hidden={hidden} toggle={toggle} />
            <DotPlot user={user} overlap={overlap} methods={shown} />
          </div>
          <PersonErrors values={view.mae} methods={shown} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

function Headline({ overlap, data }: { overlap: string; data: ReplayData }) {
  const all = data.overall[overlap], base = all["Movie average"].rmse;
  const rows = Object.entries(all).filter(([m]) => m !== "Movie average" && !LEFT_OUT.has(m)).map(([m, v]) => [m, v.rmse - base, v] as const);
  const span = Math.max(0.07, ...rows.map(r => Math.abs(r[1])));
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      <motion.div key={overlap} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-amber-400/[0.12] via-transparent to-violet-500/[0.1] p-7 ring-1 ring-inset ring-white/[0.08]">
        <p className="text-xs font-medium uppercase tracking-wider text-amber-200/70">{overlap} overlap · what the slides show</p>
        <h3 className="mt-2 text-2xl font-semibold tracking-tight text-white">{STORY[overlap].title}</h3>
        <p className="mt-3 text-sm leading-relaxed text-neutral-300">{STORY[overlap].body}</p>
        <p className="mt-4 text-xs text-neutral-500">All 433,618 hidden ratings of 25,438 test users · seed 2020 · same numbers as the saved runs</p>
      </motion.div>
      <div className="rounded-2xl bg-gradient-to-b from-white/[0.05] to-white/[0.01] p-6 ring-1 ring-inset ring-white/[0.06]">
        <div className="mb-4 flex items-baseline justify-between">
          <h3 className="font-medium text-white">RMSE change vs the movie average</h3>
          <span className="font-mono text-xs text-neutral-500">movie avg = {base.toFixed(4)}</span>
        </div>
        <div className="space-y-2">
          {rows.map(([m, d, v], k) => (
            <div key={m} className="grid grid-cols-[110px_1fr_118px] items-center gap-3 text-sm">
              <span className="truncate text-neutral-300">{SHORT[m]}</span>
              <div className="relative h-3">
                <span className="absolute left-1/2 top-[-4px] h-5 w-px bg-white/25" />
                <motion.span className={cn("absolute top-0 h-3 rounded-full", d < 0 ? "bg-emerald-400" : "bg-rose-400")}
                  initial={{ width: 0, left: "50%" }} animate={{ width: `${(Math.abs(d) / span) * 50}%`, left: d < 0 ? `${50 - (Math.abs(d) / span) * 50}%` : "50%" }}
                  transition={{ duration: 0.6, delay: k * 0.04 }} />
              </div>
              <span className={cn("text-right font-mono text-xs", d < 0 ? "text-emerald-300" : "text-rose-300")}>{d > 0 ? "+" : "−"}{Math.abs(d).toFixed(4)} <span className="text-neutral-500">({v.rmse.toFixed(4)})</span></span>
            </div>
          ))}
        </div>
        <div className="mt-3 flex justify-between text-[11px] text-neutral-500"><span>← better than movie average</span><span>worse →</span></div>
      </div>
    </div>
  );
}

function Legend({ methods, hidden, toggle }: { methods: string[]; hidden: Set<string>; toggle: (m: string) => void }) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-2">
      <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-xs text-white">
        <span className="h-3 w-3 rounded-full border-2 border-white" /> real rating
      </span>
      {methods.map(m => (
        <button key={m} onClick={() => toggle(m)}
          className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition",
            hidden.has(m) ? "border-white/5 text-neutral-600" : "border-white/10 text-neutral-200 hover:bg-white/5")}>
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: hidden.has(m) ? "#404040" : MODEL_COLORS[m] }} />{SHORT[m]}
        </button>
      ))}
    </div>
  );
}

function DotPlot({ user, overlap, methods }: { user: ReplayUser; overlap: string; methods: string[] }) {
  const x = (v: number) => `${((Math.min(5, Math.max(1, v)) - 1) / 4) * 100}%`;
  const pred = user.by_overlap[overlap].pred;
  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[minmax(160px,240px)_1fr] gap-4 pb-1 text-[11px] text-neutral-500">
        <span />
        <div className="relative h-4">{[1, 2, 3, 4, 5].map(s => <span key={s} className="absolute -translate-x-1/2 font-mono" style={{ left: x(s) }}>{s}★</span>)}</div>
      </div>
      {user.movies.map((m, k) => {
        const best = methods.length ? methods.reduce((b, cur) => Math.abs(pred[cur][k] - m.actual) < Math.abs(pred[b][k] - m.actual) ? cur : b, methods[0]) : null;
        return (
          <div key={k} className="grid grid-cols-[minmax(160px,240px)_1fr] items-center gap-4 rounded-lg px-1 py-2 hover:bg-white/[0.02]">
            <div>
              <div className="line-clamp-1 text-sm text-neutral-200" title={m.title}>{cleanTitle(m.title)}</div>
              <div className="flex items-center gap-1.5 text-[11px] text-neutral-500"><StarRow value={m.actual} size={10} />
                {best && <>closest: <span style={{ color: MODEL_COLORS[best] }}>{SHORT[best]}</span></>}</div>
            </div>
            <div className="relative h-8">
              <div className="absolute inset-x-0 top-1/2 h-px bg-white/10" />
              {[1, 2, 3, 4, 5].map(s => <span key={s} className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-white/15" style={{ left: x(s) }} />)}
              <span className="absolute top-1/2 z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[2.5px] border-white shadow-[0_0_14px_rgba(255,255,255,.5)]"
                style={{ left: x(m.actual) }} title={`real: ${m.actual}★`} />
              {methods.map((meth, j) => (
                <motion.span key={meth} initial={{ left: x(3), opacity: 0 }} animate={{ left: x(pred[meth][k]), opacity: 1 }} transition={{ delay: 0.03 * j, type: "spring", bounce: 0.2 }}
                  className="absolute top-1/2 z-20 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-neutral-950"
                  style={{ background: MODEL_COLORS[meth] }} title={`${meth}: ${pred[meth][k].toFixed(2)}`} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PersonErrors({ values, methods }: { values: Record<string, number>; methods: string[] }) {
  const rows = methods.map(m => [m, values[m]] as const).sort((a, b) => a[1] - b[1]);
  const hi = Math.max(...rows.map(r => r[1]), 0.01);
  return (
    <div className="h-fit rounded-2xl bg-gradient-to-b from-white/[0.05] to-white/[0.01] p-5 ring-1 ring-inset ring-white/[0.06]">
      <h3 className="font-medium text-white">Error for this person</h3>
      <p className="mb-4 text-xs text-neutral-500">Mean absolute error over their hidden movies, lower is better. With ~9 ratings per person this is noisy; judge models on the totals above.</p>
      <div className="space-y-2.5">
        {rows.map(([m, v], k) => (
          <div key={m} className="grid grid-cols-[92px_1fr_44px] items-center gap-2 text-sm">
            <span className="flex items-center gap-1 truncate text-neutral-300">{k === 0 && <Trophy className="h-3 w-3 text-amber-400" />}{SHORT[m]}</span>
            <div className="h-2 rounded-full bg-white/[0.05]">
              <motion.div className="h-full rounded-full" style={{ background: MODEL_COLORS[m] }} initial={{ width: 0 }}
                animate={{ width: `${Math.max(4, (v / hi) * 100)}%` }} transition={{ duration: 0.6, delay: k * 0.04 }} />
            </div>
            <span className="text-right font-mono text-xs text-neutral-300">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
