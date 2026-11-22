import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

// Shimmer button: a light sweep across a dark pill.
export function ShimmerButton({ children, onClick, disabled, className }: { children: ReactNode; onClick?: () => void; disabled?: boolean; className?: string }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className={cn("inline-flex h-11 animate-shimmer items-center justify-center gap-2 rounded-full border border-white/15 px-6 text-sm font-medium text-white transition",
        "bg-[linear-gradient(110deg,#0a0a0a,45%,#2a2540,55%,#0a0a0a)] bg-[length:200%_100%] hover:border-white/30 focus:outline-none disabled:cursor-default disabled:opacity-60", className)}>
      {children}
    </button>
  );
}
