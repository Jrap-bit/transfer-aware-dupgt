import { motion } from "framer-motion";
import { Cpu, Fingerprint, FlaskConical, GitCompareArrows, Layers, ShieldCheck, Sigma } from "lucide-react";
import type { ReactNode } from "react";
import { GlowCard } from "../components/ui/glow-card";
import { EvervaultCard } from "../components/ui/evervault";
import { Meteors } from "../components/ui/meteors";
import { cn } from "../lib/utils";

const Kicker = ({ icon: Icon, children }: { icon: typeof Cpu; children: ReactNode }) => (
  <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-neutral-500"><Icon className="h-4 w-4 text-violet-300" />{children}</p>
);
const Big = ({ children, className }: { children: ReactNode; className?: string }) => (
  <p className={cn("mt-2 bg-gradient-to-br from-white to-neutral-400 bg-clip-text text-4xl font-semibold tracking-tight text-transparent", className)}>{children}</p>
);
const Body = ({ children }: { children: ReactNode }) => <p className="mt-3 text-sm leading-relaxed text-neutral-400">{children}</p>;
const rise = (i: number) => ({ initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: i * 0.07 } });

const STEPS = [
  ["Fold-in", "Ridge regression finds the 10-number book taste u that explains your stars."],
  ["Two bridges", "Loved books (4–5★) and disliked books (1–3★) each build a personal 10×10 bridge."],
  ["Gate α", "Mixes the bridges per movie. The semantic gate also reads Jina text similarity."],
  ["Fallback w", "Pulls toward the movie's average when the model is unsure about you."],
];

const SIG = [
  ["Uncertainty blend vs movie average", "1%", "−0.029 / −0.032 / −0.030", "Significant (p < 0.001)", "good"],
  ["Uncertainty blend vs movie average", "10%", "−0.041 / −0.041 / −0.041", "Significant (p < 0.001)", "good"],
  ["Semantic gate vs DUPGT", "10%", "−0.005 / −0.004 / −0.004", "Significant, small", "good"],
  ["Uncertainty vs constant blend", "1% & 10%", "about −0.001 on all 6", "Significant, tiny", "meh"],
  ["LLM teacher vs shuffled teacher", "1% & 10%", "within ±0.0001", "No real difference", "none"],
];

export function Trust() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <motion.div {...rise(0)} className="md:col-span-2">
        <EvervaultCard className="h-full p-7">
          <Kicker icon={Fingerprint}>Same models as the slides</Kicker>
          <Big className="text-5xl">byte-for-byte</Big>
          <Body>All 24 demo models (1% and 10% overlap) were retrained from the kept book/movie factors with the original seeds. Each checkpoint's SHA-256 matches the run behind the reported numbers, and so do the fitted blend weights.</Body>
          <p className="mt-4 font-mono text-[11px] text-neutral-600">DUPGT checkpoint sha256 3b863d60…5ee2a6697 (original) = 3b863d60…5ee2a6697 (demo)</p>
        </EvervaultCard>
      </motion.div>

      <motion.div {...rise(1)}>
        <GlowCard className="h-full p-7">
          <Kicker icon={Sigma}>New-user scoring checked</Kicker>
          <Big>r = 0.97</Big>
          <Body>On 3,000 real test users, a taste vector rebuilt from book ratings alone reproduces the learned one's predictions. MAE 0.761 vs 0.758.</Body>
        </GlowCard>
      </motion.div>

      <motion.div {...rise(2)} className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-gradient-to-br from-neutral-900 to-neutral-950 p-7">
        <Meteors number={16} />
        <Kicker icon={Cpu}>Runs on this laptop</Kicker>
        <Big>&lt; 0.1 s</Big>
        <Body>Scores 5,598 movies with all seven models at once. No GPU, no internet. The LLM runs locally in LM Studio.</Body>
      </motion.div>

      <motion.div {...rise(3)} className="rounded-2xl border border-white/[0.08] bg-neutral-950 p-7 md:col-span-2">
        <Kicker icon={Layers}>How a brand-new person is scored</Kicker>
        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          {STEPS.map(([t, d], i) => (
            <div key={t} className="relative rounded-xl bg-white/[0.03] p-4 ring-1 ring-inset ring-white/[0.06]">
              <span className="font-mono text-xs text-amber-300/80">0{i + 1}</span>
              <p className="mt-1 font-medium text-white">{t}</p>
              <p className="mt-1.5 text-[13px] leading-snug text-neutral-400">{d}</p>
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div {...rise(4)} className="relative rounded-2xl border border-white/[0.08] bg-neutral-950 p-7 md:col-span-2">
        <div className="absolute inset-x-16 -top-px h-px bg-gradient-to-r from-transparent via-amber-400/70 to-transparent" />
        <Kicker icon={GitCompareArrows}>What the demo shows honestly</Kicker>
        <Big>level &gt; genre</Big>
        <Body>These models predict star ratings, so they mostly learn how generous a rater you are and which books to trust. Genre taste is only a weak signal. The LLM teacher changed error by only about ±0.0001, so its column looks like the shuffled control. That is the real result.</Body>
      </motion.div>

      <motion.div {...rise(5)} className="rounded-2xl bg-gradient-to-b from-violet-500/[0.12] to-transparent p-7 ring-1 ring-inset ring-violet-400/20">
        <Kicker icon={ShieldCheck}>Real test users tab</Kicker>
        <Big>saved files</Big>
        <Body>Tab 2 reads the original prediction files of the reported runs. Nothing there is recomputed for the demo.</Body>
      </motion.div>
      <motion.div {...rise(6)} className="rounded-2xl border border-white/[0.08] bg-neutral-950 p-7 md:col-span-3">
        <Kicker icon={FlaskConical}>Are the results significant? Paired bootstrap over 25,438 test users, 2,000 resamples, per seed</Kicker>
        <div className="mt-5 overflow-hidden rounded-xl ring-1 ring-inset ring-white/[0.06]">
          <table className="w-full text-sm">
            <thead><tr className="bg-white/[0.03] text-left text-[11px] uppercase tracking-wider text-neutral-500">
              <th className="px-4 py-2.5 font-medium">Comparison</th><th className="px-4 py-2.5 font-medium">Overlap</th>
              <th className="px-4 py-2.5 font-medium">RMSE change per seed</th><th className="px-4 py-2.5 font-medium">Verdict</th></tr></thead>
            <tbody>
              {SIG.map(([c, o, d, v, tone]) => (
                <tr key={c + o} className="border-t border-white/[0.05]">
                  <td className="px-4 py-2.5 text-neutral-200">{c}</td><td className="px-4 py-2.5 font-mono text-neutral-400">{o}</td>
                  <td className="px-4 py-2.5 font-mono text-neutral-300">{d}</td>
                  <td className="px-4 py-2.5"><span className={cn("rounded-full px-2.5 py-0.5 text-xs",
                    tone === "good" ? "bg-emerald-400/10 text-emerald-300" : tone === "meh" ? "bg-amber-400/10 text-amber-300" : "bg-white/5 text-neutral-400")}>{v}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-neutral-500">Negative = first model better. At 1% overlap the fallback recovers about three quarters of the gain full DUPGT gets with 100× more shared users.</p>
      </motion.div>
    </div>
  );
}
