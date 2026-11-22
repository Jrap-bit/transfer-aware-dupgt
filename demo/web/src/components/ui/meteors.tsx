import { useMemo } from "react";

// Aceternity "meteors": thin streaks falling diagonally across a card.
export function Meteors({ number = 14 }: { number?: number }) {
  const items = useMemo(() => Array.from({ length: number }, () => ({
    left: Math.floor(Math.random() * 700 - 300), delay: Math.random() * 1.2 + 0.2, duration: Math.floor(Math.random() * 6 + 3),
  })), [number]);
  return (
    <>
      <style>{`@keyframes meteor{0%{transform:rotate(215deg) translateX(0);opacity:1}70%{opacity:1}100%{transform:rotate(215deg) translateX(-600px);opacity:0}}`}</style>
      {items.map((m, i) => (
        <span key={i} className="pointer-events-none absolute top-0 h-0.5 w-0.5 rotate-[215deg] rounded-full bg-slate-400 shadow-[0_0_0_1px_#ffffff10]
          before:absolute before:top-1/2 before:h-px before:w-[60px] before:-translate-y-1/2 before:bg-gradient-to-r before:from-slate-400 before:to-transparent before:content-['']"
          style={{ left: m.left, animation: `meteor ${m.duration}s linear ${m.delay}s infinite` }} />
      ))}
    </>
  );
}
