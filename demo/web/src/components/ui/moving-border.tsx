import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

// Button with a rotating conic-gradient border (Aceternity "moving border").
export function MovingBorderButton({ children, onClick, disabled, className }:
  { children: ReactNode; onClick?: () => void; disabled?: boolean; className?: string }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className={cn("group relative inline-flex overflow-hidden rounded-full p-[1.5px] disabled:opacity-60", className)}>
      <span className="absolute inset-[-1000%] animate-[spin_3s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#f59e0b_0%,#8b5cf6_40%,#0a0a0a_60%,#f59e0b_100%)]" />
      <span className="relative inline-flex h-full w-full items-center justify-center gap-2 rounded-full bg-neutral-950 px-5 py-2.5 text-sm font-medium text-white backdrop-blur-3xl transition group-hover:bg-neutral-900">
        {children}
      </span>
    </button>
  );
}
