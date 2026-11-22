import { motion, useScroll, useSpring, useTransform } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

// Aceternity "tracing beam": a vertical line beside the content that fills with a gradient as you scroll.
export function TracingBeam({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [h, setH] = useState(0);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 40%", "end 70%"] });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(() => setH(ref.current?.offsetHeight ?? 0));
    ro.observe(ref.current); return () => ro.disconnect();
  }, []);
  const y1 = useSpring(useTransform(scrollYProgress, [0, 0.8], [40, h]), { stiffness: 500, damping: 90 });
  const y2 = useSpring(useTransform(scrollYProgress, [0, 1], [0, h - 160]), { stiffness: 500, damping: 90 });
  return (
    <div ref={ref} className="relative xl:pl-14">
      <div className="absolute left-0 top-2 hidden xl:block">
        <motion.div className="ml-[3px] grid h-4 w-4 place-items-center rounded-full border border-white/20 bg-neutral-950"
          style={{ boxShadow: useTransform(scrollYProgress, [0, 0.02], ["0 0 0 rgba(0,0,0,0)", "0 0 14px rgba(167,139,250,.7)"]) }}>
          <motion.div className="h-2 w-2 rounded-full" style={{ backgroundColor: useTransform(scrollYProgress, [0, 0.02], ["#525252", "#a78bfa"]) }} />
        </motion.div>
        <svg viewBox={`0 0 20 ${h}`} width="20" height={h} className="ml-1 block" aria-hidden>
          <path d={`M 1 0 V ${h}`} fill="none" stroke="rgba(255,255,255,.08)" />
          <motion.path d={`M 1 0 V ${h}`} fill="none" stroke="url(#beam)" strokeWidth="1.5" />
          <defs>
            <motion.linearGradient id="beam" gradientUnits="userSpaceOnUse" x1="0" x2="0" y1={y1} y2={y2}>
              <stop stopColor="#f59e0b" stopOpacity="0" />
              <stop stopColor="#f59e0b" />
              <stop offset="0.4" stopColor="#a78bfa" />
              <stop offset="1" stopColor="#8b5cf6" stopOpacity="0" />
            </motion.linearGradient>
          </defs>
        </svg>
      </div>
      <div>{children}</div>
    </div>
  );
}
