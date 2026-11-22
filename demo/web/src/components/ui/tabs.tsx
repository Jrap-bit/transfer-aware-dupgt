import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

// Segmented control with a sliding pill (shared layout animation).
export function Tabs<T extends string>({ value, onChange, options, id, className, size = "md" }:
  { value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode }[]; id: string; className?: string; size?: "sm" | "md" }) {
  return (
    <div className={cn("inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.03] p-1", className)}>
      {options.map(o => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={cn("relative rounded-full font-medium transition-colors", size === "sm" ? "px-3 py-1 text-[13px]" : "px-4 py-1.5 text-sm",
            value === o.value ? "text-neutral-950" : "text-neutral-400 hover:text-neutral-100")}>
          {value === o.value && (
            <motion.span layoutId={id} className="absolute inset-0 rounded-full bg-white" transition={{ type: "spring", bounce: 0.2, duration: 0.5 }} />
          )}
          <span className="relative z-10 flex items-center gap-1.5 whitespace-nowrap">{o.label}</span>
        </button>
      ))}
    </div>
  );
}
