import { motion, useMotionTemplate, useMotionValue } from "framer-motion";
import { useEffect, useState, type MouseEvent, type ReactNode } from "react";

const HEX = "0123456789abcdef";
const randomHex = (n: number) => Array.from({ length: n }, () => HEX[Math.floor(Math.random() * 16)]).join("");

// Aceternity "Evervault card": random hex characters revealed around the cursor.
export function EvervaultCard({ children, className }: { children: ReactNode; className?: string }) {
  const x = useMotionValue(-300), y = useMotionValue(-300);
  const [text, setText] = useState(() => randomHex(2400));
  useEffect(() => { const t = setInterval(() => setText(randomHex(2400)), 120); return () => clearInterval(t); }, []);
  const mask = useMotionTemplate`radial-gradient(220px at ${x}px ${y}px, white, transparent)`;
  const move = (e: MouseEvent<HTMLDivElement>) => { const r = e.currentTarget.getBoundingClientRect(); x.set(e.clientX - r.left); y.set(e.clientY - r.top); };
  return (
    <div onMouseMove={move} onMouseLeave={() => { x.set(-300); y.set(-300); }}
      className={"group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-neutral-950 " + (className ?? "")}>
      <motion.div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-amber-500/70 to-violet-600/70 opacity-0 transition duration-500 group-hover:opacity-100" style={{ maskImage: mask, WebkitMaskImage: mask }} />
      <motion.div className="pointer-events-none absolute inset-0 opacity-0 mix-blend-overlay transition group-hover:opacity-100" style={{ maskImage: mask, WebkitMaskImage: mask }}>
        <p className="h-full break-words font-mono text-[11px] font-bold leading-4 text-white">{text}</p>
      </motion.div>
      <div className="relative">{children}</div>
    </div>
  );
}
