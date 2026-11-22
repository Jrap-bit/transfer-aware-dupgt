import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import type { MouseEvent, ReactNode } from "react";

// Aceternity "3D card effect": the card tilts toward the cursor and lifts its content.
export function TiltCard({ children, className }: { children: ReactNode; className?: string }) {
  const x = useMotionValue(0.5), y = useMotionValue(0.5);
  const rx = useSpring(useTransform(y, [0, 1], [10, -10]), { stiffness: 200, damping: 18 });
  const ry = useSpring(useTransform(x, [0, 1], [-12, 12]), { stiffness: 200, damping: 18 });
  const glare = useTransform([x, y] as never, ([a, b]: number[]) => `radial-gradient(circle at ${a * 100}% ${b * 100}%, rgba(255,255,255,.22), transparent 55%)`);
  const move = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    x.set((e.clientX - r.left) / r.width); y.set((e.clientY - r.top) / r.height);
  };
  return (
    <div style={{ perspective: 900 }} className={className}>
      <motion.div onMouseMove={move} onMouseLeave={() => { x.set(0.5); y.set(0.5); }}
        style={{ rotateX: rx, rotateY: ry, transformStyle: "preserve-3d" }} className="group relative rounded-xl">
        <div style={{ transform: "translateZ(30px)" }}>{children}</div>
        <motion.div className="pointer-events-none absolute inset-0 rounded-xl opacity-0 transition-opacity group-hover:opacity-100" style={{ background: glare }} />
      </motion.div>
    </div>
  );
}
