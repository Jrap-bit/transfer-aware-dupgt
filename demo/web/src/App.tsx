import { motion } from "framer-motion";
import { FlaskConical, Sparkles } from "lucide-react";
import { useState } from "react";
import { Spotlight } from "./components/ui/spotlight";
import { GridBackground } from "./components/ui/backgrounds";
import { cn } from "./lib/utils";
import { Live } from "./sections/Live";
import { Replay } from "./sections/Replay";
import { Trust } from "./sections/Trust";

type Tab = "live" | "replay" | "trust";
const TABS: { value: Tab; label: string; sub: string; icon: typeof Sparkles }[] = [
  { value: "live", label: "Live demo", sub: "try it yourself", icon: Sparkles },
  { value: "replay", label: "Real test users", sub: "results on held-out people", icon: FlaskConical },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const go = (t: Tab) => { setTab(t); window.scrollTo({ top: 0, behavior: "smooth" }); };

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="absolute inset-x-0 top-0 h-[720px] overflow-hidden">
        <GridBackground />
        <Spotlight className="-left-10 -top-40 md:-left-32 md:-top-20" fill="#a78bfa" />
        <Spotlight className="left-[55%] -top-40 [animation-delay:1.2s]" fill="#f59e0b" />
      </div>

      <header className="sticky top-0 z-40 flex justify-center px-4 pt-4">
        <nav className="flex gap-1 rounded-2xl border border-white/10 bg-neutral-950/75 p-1.5 shadow-[0_10px_40px_-10px_rgba(0,0,0,.8)] backdrop-blur-xl">
          {TABS.map(t => (
            <button key={t.value} onClick={() => go(t.value)}
              className={cn("relative flex items-center gap-3 rounded-xl px-5 py-2.5 text-left transition-colors",
                tab === t.value ? "text-neutral-950" : "text-neutral-300 hover:bg-white/[0.05] hover:text-white")}>
              {tab === t.value && <motion.span layoutId="main-nav" className="absolute inset-0 rounded-xl bg-white" transition={{ type: "spring", bounce: 0.18, duration: 0.5 }} />}
              <t.icon className="relative z-10 h-[18px] w-[18px]" />
              <span className="relative z-10">
                <span className="block text-[15px] font-semibold leading-tight">{t.label}</span>
                <span className={cn("block text-[11px] leading-tight", tab === t.value ? "text-neutral-500" : "text-neutral-500")}>{t.sub}</span>
              </span>
            </button>
          ))}
        </nav>
      </header>

      <main className="relative z-10 mx-auto max-w-[1400px] px-6 pb-24">
        <motion.div key={tab} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35 }}>
          {tab === "live" && <Live />}
          {tab === "replay" && <><Hero tab="replay" /><Replay /></>}
          {tab === "trust" && <><Hero tab="trust" /><Trust /></>}
        </motion.div>
      </main>
    </div>
  );
}

const HERO: Record<"replay" | "trust", { title: React.ReactNode; sub: string }> = {
  replay: {
    title: <>How close did each model get to <span className="bg-gradient-to-r from-sky-300 to-violet-400 bg-clip-text text-transparent">real ratings?</span></>,
    sub: "People the models never saw. Each dot is a model's prediction, and the white ring is what the person actually gave.",
  },
  trust: {
    title: <>Why this demo is <span className="bg-gradient-to-r from-emerald-300 to-sky-400 bg-clip-text text-transparent">the real thing</span></>,
    sub: "The checks behind every number you see on screen.",
  },
};

function Hero({ tab }: { tab: "replay" | "trust" }) {
  const h = HERO[tab];
  return (
    <div className="pb-10 pt-14 md:pt-16">
      <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-4xl font-semibold tracking-tight text-transparent md:text-5xl">
        {h.title}
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="mt-4 max-w-2xl text-base text-neutral-400 md:text-lg">
        {h.sub}
      </motion.p>
    </div>
  );
}
