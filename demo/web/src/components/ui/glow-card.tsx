import { motion, useMotionTemplate, useMotionValue } from "framer-motion";
import type { MouseEvent, ReactNode } from "react";
import { cn } from "../../lib/utils";

// Card with a cursor-following radial glow (Aceternity "card spotlight").
export function GlowCard({ children, className, color = "rgba(139,92,246,0.14)" }: { children: ReactNode; className?: string; color?: string }) {
  const x = useMotionValue(-400), y = useMotionValue(-400);
  const move = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    x.set(e.clientX - r.left); y.set(e.clientY - r.top);
  };
  const bg = useMotionTemplate`radial-gradient(420px circle at ${x}px ${y}px, ${color}, transparent 70%)`;
  return (
    <div onMouseMove={move} onMouseLeave={() => { x.set(-400); y.set(-400); }}
      className={cn("group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-neutral-900/50 backdrop-blur-sm", className)}>
      <motion.div className="pointer-events-none absolute inset-0 opacity-0 transition duration-300 group-hover:opacity-100" style={{ background: bg }} />
      <div className="relative">{children}</div>
    </div>
  );
}
