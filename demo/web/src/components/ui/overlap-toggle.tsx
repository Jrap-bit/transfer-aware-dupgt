import { motion } from "framer-motion";
import { Users } from "lucide-react";
import { SHARED } from "../../lib/api";
import { cn } from "../../lib/utils";

// "How many users do Books and Movies share?" — the limited-overlap axis of the study.
export function OverlapToggle({ value, onChange, options, available, id }:
  { value: string; onChange: (v: string) => void; options: string[]; available?: string[]; id: string }) {
  return (
    <div className="inline-flex items-center gap-1 rounded-2xl border border-white/10 bg-neutral-950/80 p-1 backdrop-blur">
      <span className="flex items-center gap-1.5 px-3 text-xs text-neutral-400"><Users className="h-3.5 w-3.5" />Shared users</span>
      {options.map(o => {
        const ok = !available || available.includes(o);
        return (
          <button key={o} disabled={!ok} onClick={() => onChange(o)} title={ok ? "" : "still training"}
            className={cn("relative rounded-xl px-3.5 py-1.5 text-left transition", ok ? "cursor-pointer" : "cursor-not-allowed opacity-40")}>
            {value === o && <motion.span layoutId={id} className="absolute inset-0 rounded-xl bg-gradient-to-br from-amber-300 to-amber-500" transition={{ type: "spring", bounce: 0.2, duration: 0.5 }} />}
            <span className={cn("relative z-10 block text-sm font-semibold leading-none", value === o ? "text-neutral-950" : "text-neutral-200")}>{o}</span>
            <span className={cn("relative z-10 mt-0.5 block font-mono text-[10px] leading-none", value === o ? "text-neutral-900/70" : "text-neutral-500")}>{SHARED[o]} users</span>
          </button>
        );
      })}
    </div>
  );
}
