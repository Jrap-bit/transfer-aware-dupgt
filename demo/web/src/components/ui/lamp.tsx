import { motion } from "framer-motion";
import type { ReactNode } from "react";

// Lamp-style header (after Aceternity "lamp"): a glowing bar casting a soft fan of light onto the heading.
export function Lamp({ children }: { children: ReactNode }) {
  return (
    <div className="relative isolate flex h-[240px] w-full flex-col items-center justify-end overflow-hidden">
      <motion.div initial={{ opacity: 0, scaleX: 0.4 }} whileInView={{ opacity: 1, scaleX: 1 }} viewport={{ once: true }} transition={{ duration: 0.9, ease: "easeOut" }}
        className="absolute left-1/2 top-0 h-[240px] w-[900px] -translate-x-1/2"
        style={{
          background: "conic-gradient(from 180deg at 50% 0%, transparent 125deg, rgba(245,158,11,.55) 160deg, rgba(255,255,255,.35) 180deg, rgba(139,92,246,.55) 200deg, transparent 235deg)",
          maskImage: "radial-gradient(ellipse 50% 100% at 50% 0%, black 10%, transparent 75%)",
          WebkitMaskImage: "radial-gradient(ellipse 50% 100% at 50% 0%, black 10%, transparent 75%)",
          filter: "blur(6px)",
        }} />
      <motion.div initial={{ width: 80, opacity: 0 }} whileInView={{ width: 420, opacity: 1 }} viewport={{ once: true }} transition={{ duration: 0.9, ease: "easeOut" }}
        className="absolute top-0 h-[2px] rounded-full bg-gradient-to-r from-amber-300 via-white to-violet-400 shadow-[0_0_24px_4px_rgba(255,255,255,.35)]" />
      <div className="absolute top-0 h-24 w-[360px] rounded-full bg-violet-500/25 blur-3xl" />
      <div className="relative z-20 px-4 pb-10">{children}</div>
    </div>
  );
}
