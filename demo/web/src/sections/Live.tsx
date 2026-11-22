import { AnimatePresence, motion } from "framer-motion";
import {
  BookHeart, BookX, Brain, Church, Heart, Landmark, Loader2, Search, Sparkles, Baby, Fingerprint, X, Check, ChevronDown, Wand2,
  Plus, Shuffle, ArrowDown, MousePointerClick, Laugh, Vote, Flag, Atom, Zap, Rocket, Skull, Briefcase, Star, Eye,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, MODEL_COLORS, SHORT, type Book, type LlmResult, type Persona, type Pick, type Recommendation } from "../lib/api";
import { cn, signed } from "../lib/utils";
import { TiltCard } from "../components/ui/tilt-card";
import { BackgroundGradient } from "../components/ui/background-gradient";
import { ShimmerButton } from "../components/ui/shimmer-button";
import { OverlapToggle } from "../components/ui/overlap-toggle";
import { TextGenerate } from "../components/ui/text-generate";
import { Lamp } from "../components/ui/lamp";
import { Modal } from "../components/ui/modal";
import { FallbackExplainer } from "./Fallback";
import { RateMovies } from "./RateMovies";
import { StarInput, StarRow } from "../components/ui/stars";
import { Tabs } from "../components/ui/tabs";
import { AnimatedNumber } from "../components/ui/number";
import { Poster, cleanTitle } from "../components/ui/poster";

const PERSONA_ICON: Record<string, typeof Heart> = {
  "Comedy memoirs": Laugh, "Progressive politics": Vote, "Conservative politics": Flag, "Popular science": Atom,
  "Comics & graphic novels": Zap, "Hard sci-fi": Rocket, "True crime": Fingerprint, "Young adult": Star,
  "Stephen King horror": Skull, "Thriller & mystery fan": Eye, "Business & self-improvement": Briefcase,
  "Fantasy & sci-fi reader": Wand2, "Parent reading picture books": Baby, "History buff": Landmark,
  "Romance reader": Heart, "Faith & inspiration reader": Church,
};
const SHOWN = 7;
const clamp = (v: number) => Math.max(1, Math.min(5, v));
type View = "verdicts" | "rate" | "famous" | "top" | "llm";

export function Live() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [base, setBase] = useState<Pick[]>([]);
  const [mood, setMood] = useState(0);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [overlap, setOverlap] = useState("10%");
  const [overlaps, setOverlaps] = useState<string[]>(["10%"]);
  const [revealed, setRevealed] = useState(false);
  const [view, setView] = useState<View>("verdicts");
  const [drill, setDrill] = useState<string | null>(null);
  // "Rate a movie" starts with the reader's own closest-matching films (filled in RateMovies); kept here so tab switches don't reset it
  const [rateItems, setRateItems] = useState<number[]>([]);
  const [rateFor, setRateFor] = useState<string | null>(null);
  const [keepFilms, setKeepFilms] = useState(false);
  const results = useRef<HTMLDivElement>(null);
  const [llm, setLlm] = useState<LlmState>({ res: null, busy: false, started: 0, key: "" });
  const picks = useMemo(() => base.map(p => ({ ...p, stars: clamp(p.stars + mood) })), [base, mood]);
  const shelfKey = picks.map(p => `${p.item}:${p.stars}`).join(",");
  const askLlm = async () => {
    if (!picks.length || llm.busy) return;
    const key = shelfKey;
    setLlm({ res: null, busy: true, started: performance.now(), key });
    try { const res = await api.llm(picks, overlap); setLlm(s => ({ ...s, res, busy: false })); }
    catch (e) { setLlm(s => ({ ...s, res: { ok: false, error: String(e) }, busy: false })); }
  };

  useEffect(() => { api.personas().then(setPersonas); api.status().then(s => setOverlaps(s.overlaps)).catch(() => {}); }, []);
  useEffect(() => {
    if (!picks.length) { setRec(null); setRevealed(false); return; }
    let live = true; setLoading(true);
    const t = setTimeout(() => api.recommend(picks, "rating", overlap).then(r => { if (live) { setRec(r); setOverlaps(r.overlaps); } })
      .finally(() => live && setLoading(false)), 120);
    return () => { live = false; clearTimeout(t); };
  }, [picks, overlap]);

  const choose = (p: Persona) => { setActive(p.name); setMood(0); setBase(p.picks.map(x => ({ ...x }))); };
  const setStars = (item: number, stars: number) => setBase(b => b.map(p => p.item === item ? { ...p, stars: clamp(stars - mood) } : p));
  const remove = (item: number) => setBase(b => b.filter(p => p.item !== item));
  const add = (book: Book) => { setActive(null); setBase(b => b.some(p => p.item === book.item) ? b : [...b, { item: book.item, title: book.title, stars: 5 }]); };
  const reveal = () => { setRevealed(true); setTimeout(() => results.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80); };

  return (
    <div>
      {/* ---------- 1. the input is the hero ---------- */}
      <section className="mx-auto max-w-3xl pb-8 pt-16 text-center md:pt-20">
        <h1 className="bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-4xl font-semibold tracking-tight text-transparent md:text-6xl">
          <TextGenerate words="Your books know your movies." highlight={[3, 4]} />
        </h1>
        <div className="mt-10 text-left"><BookSearch onAdd={add} /></div>
        <Suggestions onAdd={add} taken={base.map(b => b.item)} />
        <Readers personas={personas} active={active} onChoose={choose} />
        <p className="mt-5 font-mono text-[11px] tracking-wide text-neutral-600">477,781 books to pick from · 5,598 movies ranked · 34,413 movies you can rate</p>
      </section>

      <AnimatePresence>
        {base.length > 0 && (
          <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mx-auto max-w-5xl">
            <div className="grid gap-4 md:grid-cols-[1fr_1fr_240px]">
              <Shelf picks={picks} onStars={setStars} onRemove={remove} />
              <div className="space-y-3">
                <MoodSlider value={mood} onChange={setMood} disabled={!base.length} />
                <ShimmerButton onClick={reveal} className="h-14 w-full text-base">
                  {revealed ? "Jump to predictions" : "Show my predictions"} <ArrowDown className="h-4 w-4" />
                </ShimmerButton>
                <p className="text-center text-[11px] text-neutral-500">{picks.filter(p => p.stars >= 4).length} loved · {picks.filter(p => p.stars < 4).length} not for me</p>
              </div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* ---------- 2. results: the focus once the shelf is done ---------- */}
      <AnimatePresence>
        {revealed && rec && (
          <motion.section ref={results} initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }} className="scroll-mt-24 pt-20">
            <Lamp>
              <h2 className="bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-center text-3xl font-semibold tracking-tight text-transparent md:text-5xl">
                What the models predict for you
              </h2>
            </Lamp>
            <div className="relative z-10 -mt-6 flex flex-wrap items-center justify-center gap-3">
              <OverlapToggle id="live-overlap" value={overlap} onChange={setOverlap} options={["1%", "10%"]} available={overlaps} />
              {loading && <Loader2 className="h-4 w-4 animate-spin text-neutral-500" />}
            </div>
            <p className="mt-3 text-center text-sm text-neutral-400">
              {overlap === "1%" ? "Trained with only 890 shared users. Watch the fallback earn its keep." : "Trained with 10% of the shared users (8,903 people)."}
            </p>

            <Headlines rec={rec} onExplain={() => setDrill(rec.fallback[0]?.title ?? null)} />

            <div className="mt-10 flex justify-center">
              <Tabs id="result-view" value={view} onChange={setView} options={[
                { value: "verdicts", label: "Model verdicts" }, { value: "rate", label: "Rate a movie" }, { value: "famous", label: "15 famous movies" },
                { value: "top", label: "Highest predicted" }, { value: "llm", label: <>{llm.busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}Ask the LLM{llm.res?.ok && !llm.busy && view !== "llm" && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />}</> },
              ]} />
            </div>
            <div className="mt-6">
              <AnimatePresence mode="wait">
                <motion.div key={view} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
                  {view === "verdicts" && <Verdicts rec={rec} />}
                  {view === "rate" && <RateMovies picks={picks} overlap={overlap} items={rateItems} setItems={setRateItems} reader={active ?? "your shelf"} loadedFor={rateFor} setLoadedFor={setRateFor} keep={keepFilms} setKeep={setKeepFilms} />}
                  {view === "famous" && <Heatmap rec={rec} onPick={setDrill} />}
                  {view === "top" && <TopPicks rec={rec} picks={picks} />}
                  {view === "llm" && <AskLlm state={llm} onAsk={askLlm} canAsk={!!picks.length} stale={!!llm.res && llm.key !== shelfKey} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <Modal open={!!drill} onClose={() => setDrill(null)} title="How the fallback decided this rating"
        sub="Five bridges vote. The more they disagree about you, the more the final rating leans on the movie's average.">
        {rec && drill && <FallbackExplainer rec={rec} initial={drill} />}
      </Modal>
    </div>
  );
}

/* ---------------- reader presets ---------------- */
function Readers({ personas, active, onChoose }: { personas: Persona[]; active: string | null; onChoose: (p: Persona) => void }) {
  const [more, setMore] = useState(false);
  const list = more ? personas : personas.slice(0, SHOWN);
  const current = personas.find(p => p.name === active);
  return (
    <div className="mt-5">
      <div className="flex flex-wrap items-center justify-center gap-1.5 text-xs">
        <span className="mr-1 text-neutral-500">or load a reader:</span>
        <AnimatePresence initial={false}>
          {list.map(p => {
            const Icon = PERSONA_ICON[p.name] ?? BookHeart;
            return (
              <motion.button key={p.name} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
                onClick={() => onChoose(p)} title={p.note || undefined}
                className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 transition",
                  active === p.name ? "border-amber-400/50 bg-amber-400/10 text-amber-200" : "border-white/10 text-neutral-400 hover:border-white/25 hover:text-neutral-200")}>
                <Icon className="h-3 w-3" />{p.name}
                {p.note && <Sparkles className="h-3 w-3 text-violet-300" />}
              </motion.button>
            );
          })}
        </AnimatePresence>
        {personas.length > SHOWN && (
          <button onClick={() => setMore(m => !m)} className="rounded-full px-2 py-1 text-neutral-500 underline-offset-4 hover:text-neutral-200 hover:underline">
            {more ? "fewer" : `+ ${personas.length - SHOWN} more`}
          </button>
        )}
      </div>
      <AnimatePresence>
        {current?.note && (
          <motion.p key={current.name} initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-violet-500/10 px-3 py-1 text-xs text-violet-200 ring-1 ring-inset ring-violet-400/20">
            <Sparkles className="h-3 w-3" />{current.note}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ---------------- headline cards ---------------- */
function Headlines({ rec, onExplain }: { rec: Recommendation; onExplain: () => void }) {
  const col = (m: string) => rec.columns.find(c => c.model === m);
  const sem = col("+ Semantic gate") ?? col("DUPGT");
  const blend = col("Uncertainty blend");
  const shift = sem?.summary.mean_shift ?? 0, alpha = sem?.summary.mean_alpha_high ?? 0.5, w = blend?.summary.mean_trust ?? null;
  const kind = shift < -0.15 ? "a tough critic" : shift > 0.15 ? "a generous rater" : "an average rater";
  return (
    <div className="mx-auto mt-10 grid max-w-5xl gap-4 md:grid-cols-3">
      <Headline kicker="You look like" accent="from-rose-400/20" big={kind}
        detail={<><AnimatedNumber value={shift} signed />★ vs the average viewer, according to the semantic gate</>} />
      <Headline kicker="The gate listens to" accent="from-amber-400/20"
        big={<><AnimatedNumber value={alpha * 100} digits={0} suffix="%" /> loved books</>}
        detail={<><AnimatedNumber value={(1 - alpha) * 100} digits={0} suffix="%" /> to the books you didn't enjoy</>} />
      <Headline kicker="The fallback trusts your bridge" accent="from-yellow-300/20"
        big={w == null ? "…" : <>w = <AnimatedNumber value={w} /></>}
        detail={<button onClick={onExplain} className="inline-flex items-center gap-1 text-yellow-200 underline decoration-yellow-200/30 underline-offset-4 hover:decoration-yellow-200">
          <MousePointerClick className="h-3.5 w-3.5" />see how it decides</button>} />
    </div>
  );
}

function Headline({ kicker, big, detail, accent }: { kicker: string; big: React.ReactNode; detail: React.ReactNode; accent: string }) {
  return (
    <motion.div layout className={cn("rounded-2xl bg-gradient-to-br to-transparent p-6 ring-1 ring-inset ring-white/10", accent)}>
      <p className="text-xs font-medium uppercase tracking-wider text-neutral-400">{kicker}</p>
      <p className="mt-2 text-3xl font-semibold tracking-tight text-white">{big}</p>
      <p className="mt-2 text-sm text-neutral-400">{detail}</p>
    </motion.div>
  );
}

/* ---------------- suggestions ---------------- */
function Suggestions({ onAdd, taken }: { onAdd: (b: Book) => void; taken: number[] }) {
  const [books, setBooks] = useState<Book[]>([]);
  const load = () => api.suggest(10).then(setBooks);
  useEffect(() => { load(); }, []);
  const shown = books.filter(b => !taken.includes(b.item)).slice(0, 7);
  return (
    <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
      <span className="text-xs text-neutral-500">Try adding</span>
      <AnimatePresence mode="popLayout">
        {shown.map(b => (
          <motion.button key={b.item} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
            onClick={() => onAdd(b)}
            className="group inline-flex max-w-[260px] items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[13px] text-neutral-200 transition hover:border-amber-400/50 hover:bg-amber-400/10">
            <Plus className="h-3.5 w-3.5 shrink-0 text-neutral-500 group-hover:text-amber-300" /><span className="truncate">{cleanTitle(b.title)}</span>
          </motion.button>
        ))}
      </AnimatePresence>
      <button onClick={load} className="grid h-8 w-8 place-items-center rounded-full border border-white/10 text-neutral-400 transition hover:rotate-180 hover:text-white" title="More suggestions">
        <Shuffle className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

/* ---------------- step 1 ---------------- */
const PLACEHOLDERS = ["Search 478,000 books…", "Try “Harry Potter”", "Try “Gone Girl”", "Try “Pride and Prejudice”", "Try “The Martian”"];

function BookSearch({ onAdd }: { onAdd: (b: Book) => void }) {
  const [q, setQ] = useState(""), [hits, setHits] = useState<Book[]>([]), [open, setOpen] = useState(false), [ph, setPh] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => { const t = setInterval(() => setPh(i => (i + 1) % PLACEHOLDERS.length), 2600); return () => clearInterval(t); }, []);
  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const t = setTimeout(() => api.search(q).then(h => { setHits(h); setOpen(true); }), 150);
    return () => clearTimeout(t);
  }, [q]);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close); return () => document.removeEventListener("mousedown", close);
  }, []);
  return (
    <div ref={box} className="relative">
      <div className="relative flex items-center rounded-2xl border border-white/15 bg-neutral-900/80 shadow-[0_0_0_4px_rgba(139,92,246,.06),0_20px_60px_-20px_rgba(139,92,246,.45)] transition focus-within:border-violet-400/60 focus-within:shadow-[0_0_0_4px_rgba(139,92,246,.15),0_20px_60px_-15px_rgba(139,92,246,.6)]">
        <Search className="ml-5 h-5 w-5 text-neutral-400" />
        <input value={q} onChange={e => setQ(e.target.value)} onFocus={() => hits.length && setOpen(true)}
          className="peer h-16 w-full bg-transparent px-4 text-lg text-white outline-none" />
        {!q && (
          <AnimatePresence mode="wait">
            <motion.span key={ph} initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -8, opacity: 0 }} transition={{ duration: 0.25 }}
              className="pointer-events-none absolute left-14 text-lg text-neutral-500">{PLACEHOLDERS[ph]}</motion.span>
          </AnimatePresence>
        )}
      </div>
      <AnimatePresence>
        {open && hits.length > 0 && (
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
            className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-white/10 bg-neutral-900/95 p-1 shadow-2xl backdrop-blur">
            {hits.map(h => (
              <button key={h.item} onClick={() => { onAdd(h); setQ(""); setOpen(false); }}
                className="flex w-full flex-col rounded-lg px-3 py-2 text-left hover:bg-white/5">
                <span className="line-clamp-1 text-sm text-neutral-100">{h.title}</span>
                <span className="text-xs text-neutral-500">{h.category || "Books"} · {h.n_ratings.toLocaleString()} ratings</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MoodSlider({ value, onChange, disabled }: { value: number; onChange: (v: number) => void; disabled: boolean }) {
  const labels = ["Very harsh", "Harsh", "As picked", "Generous", "Very generous"];
  return (
    <div className={cn("rounded-xl border border-white/10 bg-white/[0.02] p-4", disabled && "opacity-50")}>
      <div className="mb-3 flex items-center justify-between text-sm">
        <span className="font-medium text-neutral-200">Rater mood</span>
        <span className="rounded-full bg-white/5 px-2.5 py-0.5 text-xs text-neutral-300">{labels[value + 2]}{value !== 0 && ` · every star ${value > 0 ? "+" : ""}${value}`}</span>
      </div>
      <input type="range" min={-2} max={2} step={1} value={value} disabled={disabled} onChange={e => onChange(+e.target.value)}
        className="w-full cursor-pointer accent-amber-400" />
      <div className="mt-1 flex justify-between text-[11px] text-neutral-500"><span>Tough critic</span><span>Loves everything</span></div>
    </div>
  );
}

function Shelf({ picks, onStars, onRemove }: { picks: Pick[]; onStars: (i: number, s: number) => void; onRemove: (i: number) => void }) {
  const liked = picks.filter(p => p.stars >= 4), disliked = picks.filter(p => p.stars < 4);
  const lane = (title: string, sub: string, items: Pick[], tone: "liked" | "disliked", Icon: typeof BookHeart) => (
    <div className={cn("flex min-h-[120px] flex-col rounded-xl border p-3", tone === "liked" ? "border-amber-400/15 bg-gradient-to-b from-amber-400/[0.07] to-transparent" : "border-white/[0.07] bg-gradient-to-b from-white/[0.03] to-transparent")}>
      <div className="mb-2 flex items-center gap-2 px-1">
        <Icon className={cn("h-4 w-4", tone === "liked" ? "text-amber-400" : "text-neutral-400")} />
        <span className="text-sm font-medium text-neutral-100">{title}</span>
        <span className="ml-auto text-[11px] text-neutral-500">{sub}</span>
      </div>
      <div className="flex flex-col gap-1.5">
        <AnimatePresence initial={false}>
          {items.map(p => (
            <motion.div key={p.item} layout initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }}
              className="group flex items-center gap-2 rounded-lg bg-neutral-950/60 px-2.5 py-1.5">
              <span className="line-clamp-1 flex-1 text-[13px] text-neutral-200" title={p.title}>{cleanTitle(p.title)}</span>
              <StarInput value={p.stars} onChange={s => onStars(p.item, s)} tone={tone} />
              <button onClick={() => onRemove(p.item)} className="text-neutral-600 opacity-0 transition hover:text-neutral-200 group-hover:opacity-100"><X className="h-3.5 w-3.5" /></button>
            </motion.div>
          ))}
        </AnimatePresence>
        {!items.length && <span className="px-1 py-3 text-xs text-neutral-600">Nothing here yet</span>}
      </div>
    </div>
  );
  return (
    <>
      {lane("Loved", "4–5★ → liked branch", liked, "liked", BookHeart)}
      {lane("Not for me", "1–3★ → disliked branch", disliked, "disliked", BookX)}
    </>
  );
}

/* ---------------- step 2 ---------------- */
function verdict(shift: number, alpha: number | null, trust: number | null, model: string) {
  if (model === "Movie average") return "Ignores your books: everyone gets the same predictions.";
  const level = shift < -0.15 ? "Sees you as a tough critic" : shift > 0.15 ? "Expects you to rate generously" : "Expects you to rate about average";
  const gate = alpha == null ? "" : alpha >= 0.6 ? ", leaning on books you loved" : alpha <= 0.4 ? ", leaning on books you disliked" : ", weighing loved and disliked books evenly";
  const tr = trust == null ? "" : trust < 0.55 ? ". Unsure about you, so it falls back toward movie averages" : ". Fairly confident in its personal guess";
  return level + gate + tr + ".";
}

function Verdicts({ rec }: { rec: Recommendation }) {
  return (
    <div className="overflow-hidden rounded-xl border border-white/[0.07]">
      <div className="hidden grid-cols-[220px_1fr_210px_170px_120px] gap-4 border-b border-white/[0.07] bg-white/[0.02] px-4 py-2.5 text-[11px] font-medium uppercase tracking-wider text-neutral-500 lg:grid">
        <span>Model</span><span>Verdict</span><span>Shift vs movie average</span><span>Liked ↔ disliked books</span><span>Trust in you</span>
      </div>
      {rec.columns.map((c, i) => {
        const s = c.summary, color = MODEL_COLORS[c.model];
        return (
          <motion.div key={c.model} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.04 }}
            className="grid items-center gap-x-4 gap-y-2 border-b border-white/[0.05] px-4 py-3.5 last:border-0 lg:grid-cols-[220px_1fr_210px_170px_120px]">
            <div className="flex items-center gap-2.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color, boxShadow: `0 0 12px ${color}` }} />
              <span className="font-medium text-white">{c.model}</span>
            </div>
            <p className="text-[13.5px] leading-snug text-neutral-400">{verdict(s.mean_shift, s.mean_alpha_high, s.mean_trust, c.model)}</p>
            <ShiftBar shift={c.model === "Movie average" ? 0 : s.mean_shift} color={color} />
            {s.mean_alpha_high == null ? <span className="text-xs text-neutral-600">n/a</span> : <GateBar alpha={s.mean_alpha_high} />}
            {s.mean_trust == null ? <span className="text-xs text-neutral-600">n/a</span> : <TrustRing w={s.mean_trust} />}
          </motion.div>
        );
      })}
    </div>
  );
}

function ShiftBar({ shift, color }: { shift: number; color: string }) {
  const max = 1.2, pct = Math.min(1, Math.abs(shift) / max) * 50;
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-2 flex-1 rounded-full bg-white/[0.06]">
        <span className="absolute left-1/2 top-[-3px] h-[14px] w-px bg-white/30" />
        <motion.span className="absolute top-0 h-2 rounded-full" style={{ background: color }}
          animate={{ left: shift < 0 ? `${50 - pct}%` : "50%", width: `${pct}%` }} transition={{ type: "spring", bounce: 0.15, duration: 0.7 }} />
      </div>
      <span className={cn("w-14 text-right font-mono text-sm", shift > 0.02 ? "text-emerald-400" : shift < -0.02 ? "text-rose-400" : "text-neutral-400")}>
        <AnimatedNumber value={shift} signed />★
      </span>
    </div>
  );
}

function GateBar({ alpha }: { alpha: number }) {
  return (
    <div>
      <div className="flex h-2 overflow-hidden rounded-full bg-neutral-700">
        <motion.span className="h-full bg-amber-400" animate={{ width: `${alpha * 100}%` }} transition={{ type: "spring", bounce: 0.15, duration: 0.7 }} />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[11px]">
        <span className="text-amber-400/90"><AnimatedNumber value={alpha * 100} digits={0} suffix="%" /></span>
        <span className="text-neutral-500"><AnimatedNumber value={(1 - alpha) * 100} digits={0} suffix="%" /></span>
      </div>
    </div>
  );
}

function TrustRing({ w }: { w: number }) {
  const r = 15, c = 2 * Math.PI * r;
  return (
    <div className="flex items-center gap-2">
      <svg width="38" height="38" viewBox="0 0 38 38" className="-rotate-90">
        <circle cx="19" cy="19" r={r} stroke="rgba(255,255,255,.08)" strokeWidth="4" fill="none" />
        <motion.circle cx="19" cy="19" r={r} stroke="#facc15" strokeWidth="4" fill="none" strokeLinecap="round" strokeDasharray={c}
          animate={{ strokeDashoffset: c * (1 - w) }} transition={{ duration: 0.7 }} />
      </svg>
      <span className="font-mono text-sm text-neutral-200">w=<AnimatedNumber value={w} /></span>
    </div>
  );
}

/* ---------------- step 3 ---------------- */
function heat(d: number) {
  const a = Math.min(1, Math.abs(d) / 0.9);
  return d >= 0 ? `rgba(16,185,129,${0.06 + 0.5 * a})` : `rgba(244,63,94,${0.06 + 0.5 * a})`;
}

function Heatmap({ rec, onPick }: { rec: Recommendation; onPick: (title: string) => void }) {
  const models = rec.columns.map(c => c.model).filter(m => m !== "Movie average");
  return (
    <div className="overflow-auto rounded-xl border border-white/[0.07]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-white/[0.02] text-[11px] uppercase tracking-wider text-neutral-500">
            <th className="px-4 py-2.5 text-left font-medium">Movie</th>
            <th className="px-3 py-2.5 text-right font-medium">Avg</th>
            {models.map(m => (
              <th key={m} className="px-3 py-2.5 text-center font-medium">
                <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full" style={{ background: MODEL_COLORS[m] }} />{SHORT[m]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rec.probe.map(p => (
            <tr key={p.title} onClick={() => onPick(p.title)} className="group cursor-pointer border-t border-white/[0.05] transition hover:bg-white/[0.03]">
              <td className="px-4 py-2 text-neutral-200">{cleanTitle(p.title)}
                <span className="ml-2 text-[11px] text-yellow-300/0 transition group-hover:text-yellow-300/80">how did the fallback decide? →</span></td>
              <td className="px-3 py-2 text-right font-mono text-neutral-500">{p.movie_avg.toFixed(2)}</td>
              {models.map(m => {
                const v = p.predicted[m]; if (!v) return <td key={m} />;
                const d = v.stars - p.movie_avg;
                return (
                  <td key={m} className="px-1.5 py-1">
                    <motion.div animate={{ backgroundColor: heat(d) }} transition={{ duration: 0.5 }}
                      className="rounded-md px-2 py-1.5 text-center font-mono text-[13px] text-white" title={`${signed(d)}★ vs average`}>
                      <AnimatedNumber value={v.stars} />
                    </motion.div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TopPicks({ rec, picks }: { rec: Recommendation; picks: Pick[] }) {
  const names = rec.columns.map(c => c.model);
  const [model, setModel] = useState(names.includes("+ Semantic gate") ? "+ Semantic gate" : names[1] ?? names[0]);
  const [rank, setRank] = useState<"rating" | "boost">("rating");
  const [boosted, setBoosted] = useState<Recommendation | null>(null);
  useEffect(() => { if (rank === "boost") api.recommend(picks, "boost").then(setBoosted); }, [rank, picks]);
  const source = rank === "boost" && boosted ? boosted : rec;
  const col = source.columns.find(c => c.model === model) ?? source.columns[0];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Tabs id="top-model" size="sm" value={model} onChange={setModel}
          options={names.map(n => ({ value: n, label: <><span className="h-1.5 w-1.5 rounded-full" style={{ background: MODEL_COLORS[n] }} />{SHORT[n]}</> }))} />
        <Tabs id="top-rank" size="sm" value={rank} onChange={setRank} options={[{ value: "rating", label: "Highest stars" }, { value: "boost", label: "Biggest personal boost" }]} />
      </div>
      <p className="rounded-lg bg-amber-400/[0.07] px-3 py-2 text-xs text-amber-100/90 ring-1 ring-inset ring-amber-400/15">Rating prediction, not topic matching: these are the movies this model expects you to <b className="font-semibold">rate highest</b> out of 5,598 well-known titles. They reflect audiences with similar rating patterns, not the subject of your books.</p>
      <p className="text-sm text-neutral-400">{col.about}{model !== "Movie average" && source.agreement[model] != null && model !== "+ Semantic gate" &&
        <span className="ml-2 rounded-full bg-white/5 px-2 py-0.5 text-xs text-neutral-300">{source.agreement[model]}/5 shared with the semantic gate</span>}</p>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <AnimatePresence mode="popLayout">
          {col.top.map((t, i) => (
            <motion.div key={model + rank + t.title} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ delay: i * 0.05 }}
              className="group">
              <TiltCard><Poster title={t.title} className="shadow-[0_20px_40px_-20px_rgba(0,0,0,.8)]" /></TiltCard>
              <div className="mt-2.5 space-y-1">
                <div className="flex items-center gap-2"><StarRow value={t.predicted} /><span className="font-mono text-sm text-white">{t.predicted.toFixed(2)}</span></div>
                <div className="text-xs text-neutral-500">avg {t.movie_avg.toFixed(2)} · {t.n.toLocaleString()} ratings
                  {model !== "Movie average" && <span className={cn("ml-1", t.boost >= 0 ? "text-emerald-400" : "text-rose-400")}>{signed(t.boost)}</span>}</div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ---------------- step 4 ---------------- */
type LlmState = { res: LlmResult | null; busy: boolean; started: number; key: string };

function AskLlm({ state, onAsk, canAsk, stale }: { state: LlmState; onAsk: () => void; canAsk: boolean; stale: boolean }) {
  const { res, busy } = state;
  const [raw, setRaw] = useState(false), [t, setT] = useState(0);
  useEffect(() => {
    if (!busy) return;
    const tick = setInterval(() => setT((performance.now() - state.started) / 1000), 100);
    return () => clearInterval(tick);
  }, [busy, state.started]);
  const ask = onAsk;
  const compare = ["DUPGT", "+ Semantic gate", "+ LLM teacher"];
  return (
    <BackgroundGradient className="p-6 md:p-8">
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-4">
        <ShimmerButton onClick={ask} disabled={busy || !canAsk}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4 text-amber-300" />}
          {busy ? "Qwen is reading your books…" : "Ask Qwen3.5-9B which movie fits"}
        </ShimmerButton>
        {busy && <span className="font-mono text-sm text-neutral-400">{t.toFixed(1)} s</span>}
        {res?.ok && <span className="font-mono text-xs text-neutral-500">answered in {res.seconds}s · {res.prompt_tokens} → {res.completion_tokens} tokens · offline</span>}
      </div>
      {stale && !busy && <p className="rounded-lg bg-amber-400/10 px-3 py-2 text-xs text-amber-200">This answer is for your previous shelf. Ask again to update it.</p>}
      {busy && (
        <div className="h-1 overflow-hidden rounded-full bg-white/5">
          <motion.div className="h-full bg-gradient-to-r from-amber-400 via-violet-500 to-amber-400 bg-[length:200%_100%] animate-shimmer" initial={{ width: "0%" }} animate={{ width: "95%" }} transition={{ duration: 18, ease: "easeOut" }} />
        </div>
      )}
      {res && !res.ok && <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-4 text-sm text-amber-200">{res.error}</div>}
      {res?.ok && (
        <div className="space-y-3">
          {res.pairs.map((p, i) => (
            <motion.div key={p.pair_id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.12 }}
              className="grid items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-3 md:grid-cols-[1fr_auto_1fr_auto]">
              <Choice title={p.a} chosen={p.llm === "A"} />
              <span className="text-center font-mono text-xs text-neutral-600">vs</span>
              <Choice title={p.b} chosen={p.llm === "B"} />
              <div className="flex flex-wrap gap-1.5 md:justify-end">
                {compare.filter(m => p.models[m]).map(m => {
                  const agree = (p.llm === "A" || p.llm === "B") && p.models[m] === p.llm;
                  return (
                    <span key={m} className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]",
                      agree ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" : "border-white/10 bg-white/[0.03] text-neutral-400")}>
                      {agree ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}{SHORT[m]}
                    </span>
                  );
                })}
              </div>
            </motion.div>
          ))}
          <p className="flex items-start gap-2 text-sm text-neutral-400"><Brain className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
            <span>The LLM reads <b className="font-medium text-neutral-200">content</b>; the models read <b className="font-medium text-neutral-200">rating patterns</b>. A ✓ means the model also predicts more stars for the LLM's pick. During training, these choices were only weak hints, which is why the LLM teacher barely changed the error.</span></p>
          <button onClick={() => setRaw(r => !r)} className="inline-flex items-center gap-1 text-xs text-neutral-500 hover:text-neutral-300">
            <ChevronDown className={cn("h-3.5 w-3.5 transition", raw && "rotate-180")} /> exact prompt and raw JSON
          </button>
          {raw && <pre className="max-h-72 overflow-auto rounded-xl bg-black/50 p-4 font-mono text-[11px] leading-relaxed text-neutral-400">{res.prompt_preview}…{"\n\n"}{JSON.stringify(res.raw_json, null, 1)}</pre>}
        </div>
      )}
      {!res && !busy && <p className="text-sm text-neutral-500">Takes about 10–15 seconds. The models in steps 2–3 never call the LLM.</p>}
    </div>
    </BackgroundGradient>
  );
}

function Choice({ title, chosen }: { title: string; chosen: boolean }) {
  return (
    <div className={cn("flex items-center gap-3 rounded-lg border p-2 transition",
      chosen ? "border-violet-400/50 bg-violet-500/10 shadow-[0_0_30px_-8px_rgba(139,92,246,.6)]" : "border-transparent opacity-60")}>
      <div className="w-10 shrink-0"><Poster title={title} className="rounded-md p-1 [&>div:last-child]:hidden" /></div>
      <span className="text-sm text-neutral-100">{cleanTitle(title)}</span>
      {chosen && <span className="ml-auto rounded-full bg-violet-500 px-2 py-0.5 text-[11px] font-medium text-white">LLM pick</span>}
    </div>
  );
}

