import { AnimatePresence, motion } from "framer-motion";
import { useState, type ReactNode } from "react";
import { cn } from "../../lib/utils";

// Aceternity "card hover effect": one highlight that slides to whichever item is hovered.
export function HoverGrid<T>({ items, render, keyOf, className, id }:
  { items: T[]; render: (item: T) => ReactNode; keyOf: (item: T) => string; className?: string; id: string }) {
  const [hovered, setHovered] = useState<string | null>(null);
  return (
    <div className={cn("grid", className)} onMouseLeave={() => setHovered(null)}>
      {items.map(item => {
        const k = keyOf(item);
        return (
          <div key={k} className="relative p-1" onMouseEnter={() => setHovered(k)}>
            <AnimatePresence>
              {hovered === k && (
                <motion.span layoutId={id} className="absolute inset-0 block rounded-2xl bg-white/[0.07]"
                  initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { duration: 0.15 } }} exit={{ opacity: 0, transition: { duration: 0.15, delay: 0.1 } }} />
              )}
            </AnimatePresence>
            <div className="relative z-10 h-full">{render(item)}</div>
          </div>
        );
      })}
    </div>
  );
}
