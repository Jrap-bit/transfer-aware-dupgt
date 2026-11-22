import { AnimatePresence, motion } from "framer-motion";
import { Film, Lock, LockOpen, Plus, Search, Shuffle, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, MODEL_COLORS, SHORT, type MovieHit, type Pick, type Rating } from "../lib/api";
import { cn, signed } from "../lib/utils";
import { cleanTitle } from "../components/ui/poster";
import { AnimatedNumber } from "../components/ui/number";

export const QUICK = [
  { group: "Political films", titles: ["Fahrenheit 9/11", "Atlas Shrugged: Part One", "Selma", "2016: Obama", "An Inconvenient Truth", "American Sniper"] },
  { group: "Family", titles: ["Frozen", "Toy Story 3"] },
  { group: "Other", titles: ["The Shining", "Pride and Prejudice", "Saving Private Ryan"] },
];
const MAIN = "+ Semantic gate";

// "Rate a movie": the task the models are actually trained for — predicted stars for one movie, for this person.
export function RateMovies({ picks, overlap, items, setItems, reader, loadedFor, setLoadedFor, keep, setKeep }:
  { picks: Pick[]; overlap: string; items: number[]; setItems: (f: (v: number[]) => number[]) => void; reader: string;
    loadedFor: string | null; setLoadedFor: (r: string) => void; keep: boolean; setKeep: (f: (v: boolean) => boolean) => void }) {
  const [data, setData] = useState<Rating | null>(null);
  const [prev, setPrev] = useState<{ reader: string; values: Record<number, Record<string, number>> } | null>(null);
  const [focus, setFocus] = useState(MAIN);   // the model whose number is shown big; click a legend entry to change it
  const last = useRef<{ reader: string; data: Rating } | null>(null);

  useEffect(() => {
    if (!items.length || !picks.length) { setData(null); return; }
    let live = true;
    api.rate(picks, items, overlap).then(r => {
      if (!live) return;
      // remember the previous reader's numbers so a switch (e.g. conservative -> progressive) shows the change
      if (last.current && last.current.reader !== reader)
        setPrev({ reader: last.current.reader, values: Object.fromEntries(last.current.data.movies.map(m => [m.item, m.predicted])) });
      last.current = { reader, data: r }; setData(r);
    });
    return () => { live = false; };
  }, [picks, items, overlap, reader]);

  // Topical suggestions: movies whose description is closest to the loved books, refreshed when the shelf changes.
  const [topical, setTopical] = useState<MovieHit[]>([]);
  const lovedKey = picks.filter(p => p.stars >= 4).map(p => p.item).join(",");
  const loadTopical = () => { if (picks.length) api.topical(picks, items, 8).then(setTopical); };
  useEffect(loadTopical, [lovedKey]);
  // Start with this reader's 4 closest films by topic. A new reader gets their own films, unless "keep these films" is on
  // (that is how you compare two readers, e.g. conservative -> progressive, on the same movies).
  useEffect(() => {
    if (!picks.length || keep || (loadedFor === reader && items.length)) return;
    let live = true;
    api.topical(picks, [], 40).then(r => { if (live) { setItems(() => r.slice(0, 4).map(m => m.item)); setLoadedFor(reader); } });
    return () => { live = false; };
  }, [reader, lovedKey, keep]);
  const add = (hit: MovieHit) => setItems(v => v.includes(hit.item) ? v : [...v, hit.item].slice(-8));
  const remove = (item: number) => setItems(v => v.filter(i => i !== item));
  const values = data ? data.movies.flatMap(m => Object.values(m.predicted)) : [];
  const lo = values.length ? Math.max(1, Math.floor((Math.min(...values) - 0.15) * 2) / 2) : 3, hi = values.length ? Math.min(5, Math.ceil((Math.max(...values) + 0.15) * 2) / 2) : 5;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;

  return (
    <div className="space-y-5">
      <div className="rounded-2xl bg-white/[0.03] p-5 ring-1 ring-inset ring-white/[0.07]">
        <div className="flex flex-wrap items-start gap-3">
          <p className="min-w-[260px] flex-1 text-sm text-neutral-300">This is the question the models are trained to answer: <span className="text-white">"if we show this person this movie, how many stars would they give it?"</span> It starts with the films closest to this reader's books.</p>
          <button onClick={() => setKeep(v => !v)} title="On: switching reader keeps these films, so you can compare two readers on the same movies"
            className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition",
              keep ? "border-violet-400/60 bg-violet-500/15 text-violet-100" : "border-white/10 text-neutral-400 hover:text-neutral-200")}>
            {keep ? <Lock className="h-3.5 w-3.5" /> : <LockOpen className="h-3.5 w-3.5" />}{keep ? "Keeping these films across readers" : "Keep these films across readers"}
          </button>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(260px,360px)_1fr] md:items-start">
          <MovieSearch onAdd={add} />
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            {QUICK.map(g => (
              <div key={g.group} className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] uppercase tracking-wider text-neutral-500">{g.group}</span>
                {g.titles.map(t => (
                  <button key={t} onClick={() => api.searchMovies(t).then(h => h[0] && add(h[0]))}
                    className="inline-flex items-center gap-1 rounded-full border border-white/10 px-2.5 py-1 text-xs text-neutral-300 transition hover:border-violet-400/50 hover:bg-violet-500/10">
                    <Plus className="h-3 w-3 text-neutral-500" />{t.replace(": Part One", "").replace("2016: Obama", "2016: Obama's America")}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="mt-4 border-t border-white/[0.06] pt-4">
          <div className="mb-2 flex items-center gap-2 text-xs">
            <Sparkles className="h-3.5 w-3.5 text-amber-300" />
            <span className="font-medium text-amber-100">Matches {reader === "your shelf" ? "your books" : reader}</span>
            <span className="text-neutral-500">· picked by text similarity to the loved books, then rated by the models</span>
            <button onClick={loadTopical} title="Other matches" className="ml-auto grid h-7 w-7 place-items-center rounded-full border border-white/10 text-neutral-400 transition hover:rotate-180 hover:text-white"><Shuffle className="h-3.5 w-3.5" /></button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <AnimatePresence mode="popLayout">
              {topical.filter(t => !items.includes(t.item)).map(t => (
                <motion.button key={t.item} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
                  onClick={() => add(t)}
                  className="group inline-flex max-w-[280px] items-center gap-1.5 rounded-full border border-amber-400/25 bg-amber-400/[0.06] px-3 py-1.5 text-[13px] text-amber-50 transition hover:border-amber-400/60 hover:bg-amber-400/15">
                  <Plus className="h-3.5 w-3.5 shrink-0 text-amber-300/70" /><span className="truncate">{cleanTitle(t.title)}</span>
                </motion.button>
              ))}
            </AnimatePresence>
            {!topical.length && <span className="text-xs text-neutral-500">Add a loved book to get matches.</span>}
          </div>
        </div>
      </div>

      {data && data.movies.length > 0 && (
        <div className="overflow-hidden rounded-2xl ring-1 ring-inset ring-white/[0.07]">
          <div className="grid grid-cols-[minmax(200px,260px)_1fr_290px] gap-4 border-b border-white/[0.07] bg-white/[0.02] px-5 py-2.5 text-[11px] font-medium uppercase tracking-wider text-neutral-500">
            <span>Movie</span>
            <div className="relative h-4">{Array.from({ length: Math.round((hi - lo) * 2) + 1 }, (_, k) => lo + k / 2).map(t => <span key={t} className="absolute -translate-x-1/2 font-mono normal-case" style={{ left: x(t) }}>{t.toFixed(1)}★</span>)}</div>
            <span className="text-right"><span style={{ color: MODEL_COLORS[focus] }}>{SHORT[focus]}</span> for {reader}</span>
          </div>
          <AnimatePresence initial={false}>
            {data.movies.map(m => {
              const main = m.predicted[focus], before = prev?.values[m.item]?.[focus];
              return (
                <motion.div key={m.item} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="group grid grid-cols-[minmax(200px,260px)_1fr_290px] items-center gap-4 border-b border-white/[0.05] px-5 py-4 last:border-0">
                  <div className="flex items-start gap-2">
                    <Film className="mt-0.5 h-4 w-4 shrink-0 text-neutral-500" />
                    <div className="min-w-0">
                      <div className="line-clamp-2 text-sm font-medium text-neutral-100">{cleanTitle(m.title)}</div>
                      <div className="text-[11px] text-neutral-500">movie average {m.movie_avg.toFixed(2)} · {m.n.toLocaleString()} ratings</div>
                    </div>
                    <button onClick={() => remove(m.item)} className="ml-auto text-neutral-600 opacity-0 transition hover:text-neutral-200 group-hover:opacity-100"><X className="h-3.5 w-3.5" /></button>
                  </div>
                  <div className="relative h-10">
                    <div className="absolute inset-x-0 top-1/2 h-px bg-white/10" />
                    <span className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-white/40" style={{ left: x(m.movie_avg) }} title="movie average" />
                    {data.models.filter(k => k !== "Movie average").map(k => (
                      <motion.span key={k} animate={{ left: x(m.predicted[k]), scale: k === focus ? 1.6 : 1, opacity: k === focus ? 1 : 0.55 }} transition={{ type: "spring", bounce: 0.2 }}
                        onClick={() => setFocus(k)}
                        className={cn("absolute top-1/2 h-2.5 w-2.5 cursor-pointer rounded-full ring-2 ring-neutral-950", k === focus && "z-10 shadow-[0_0_12px_2px_currentColor]")}
                        style={{ background: MODEL_COLORS[k], color: MODEL_COLORS[k], x: "-50%", y: "-50%" }} title={`${k}: ${m.predicted[k].toFixed(2)}`} />
                    ))}
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-2xl font-semibold text-white"><AnimatedNumber value={main} />★</div>
                    <div className="text-xs">
                      <span className={main - m.movie_avg >= 0 ? "text-emerald-400" : "text-rose-400"}>{signed(main - m.movie_avg)} vs average</span>
                      {before != null && Math.abs(main - before) > 0.005 && (
                        <span className="ml-2 rounded-full bg-violet-500/15 px-1.5 py-0.5 text-violet-200">{signed(main - before)} vs {prev!.reader}</span>
                      )}
                    </div>
                    {m.w != null && <div className="mt-0.5 text-[11px] text-neutral-500">fallback trust w = {m.w.toFixed(2)}</div>}
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
          <div className="flex flex-wrap gap-3 border-t border-white/[0.07] bg-white/[0.02] px-5 py-2.5 text-[11px] text-neutral-400">
            <span className="inline-flex items-center gap-1.5"><span className="h-3 w-px bg-white/50" />movie average</span>
            <span className="text-neutral-500">· click a model to show its stars</span>
            {data.models.filter(k => k !== "Movie average").map(k => (
              <button key={k} onClick={() => setFocus(k)}
                className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 transition",
                  k === focus ? "border-white/30 bg-white/10 text-white" : "border-transparent hover:border-white/15 hover:text-neutral-200")}>
                <span className={cn("rounded-full transition-all", k === focus ? "h-3 w-3" : "h-2 w-2")} style={{ background: MODEL_COLORS[k] }} />{SHORT[k]}
              </button>
            ))}
          </div>
        </div>
      )}
      {(!data || !data.movies.length) && <div className="grid h-32 place-items-center rounded-2xl border border-dashed border-white/10 text-sm text-neutral-500">Add a movie above.</div>}
    </div>
  );
}

function MovieSearch({ onAdd }: { onAdd: (h: MovieHit) => void }) {
  const [q, setQ] = useState(""), [hits, setHits] = useState<MovieHit[]>([]), [open, setOpen] = useState(false);
  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const t = setTimeout(() => api.searchMovies(q).then(h => { setHits(h); setOpen(true); }), 150);
    return () => clearTimeout(t);
  }, [q]);
  return (
    <div className="relative">
      <div className="flex items-center rounded-xl border border-white/10 bg-neutral-900/80 focus-within:border-violet-400/50">
        <Search className="ml-3 h-4 w-4 text-neutral-500" />
        <input value={q} onChange={e => setQ(e.target.value)} onBlur={() => setTimeout(() => setOpen(false), 150)} onFocus={() => hits.length && setOpen(true)}
          placeholder="Any movie, e.g. Fahrenheit 9/11" className="h-10 w-full bg-transparent px-3 text-sm text-white outline-none placeholder:text-neutral-500" />
      </div>
      {open && hits.length > 0 && (
        <div className="absolute z-30 mt-1 max-h-72 w-full overflow-auto rounded-xl border border-white/10 bg-neutral-900/95 p-1 shadow-2xl backdrop-blur">
          {hits.map(h => (
            <button key={h.item} onMouseDown={() => { onAdd(h); setQ(""); setOpen(false); }} className="flex w-full flex-col rounded-lg px-3 py-2 text-left hover:bg-white/5">
              <span className="line-clamp-1 text-sm text-neutral-100">{h.title}</span>
              <span className="text-xs text-neutral-500">average {h.movie_avg.toFixed(2)} · {h.n_ratings.toLocaleString()} ratings</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
