import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

// Aceternity "background gradient": an animated multi-colour glow behind a card's border.
export function BackgroundGradient({ children, className }: { children: ReactNode; className?: string }) {
  const bg = "bg-[radial-gradient(circle_farthest-side_at_0_100%,#f59e0b,transparent),radial-gradient(circle_farthest-side_at_100%_0,#8b5cf6,transparent),radial-gradient(circle_farthest-side_at_100%_100%,#ec4899,transparent),radial-gradient(circle_farthest-side_at_0_0,#38bdf8,#0a0a0a)]";
  const anim = { backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] };
  return (
    <div className="group relative p-[1.5px]">
      <motion.div animate={anim} transition={{ duration: 6, repeat: Infinity, repeatType: "reverse" }} style={{ backgroundSize: "400% 400%" }}
        className={cn("absolute inset-0 z-[1] rounded-3xl opacity-40 blur-xl transition duration-500 group-hover:opacity-70", bg)} />
      <motion.div animate={anim} transition={{ duration: 6, repeat: Infinity, repeatType: "reverse" }} style={{ backgroundSize: "400% 400%" }}
        className={cn("absolute inset-0 z-[1] rounded-3xl", bg)} />
      <div className={cn("relative z-10 rounded-[22px] bg-neutral-950", className)}>{children}</div>
    </div>
  );
}
