import { Star } from "lucide-react";
import { cn } from "../../lib/utils";

export function StarInput({ value, onChange, tone }: { value: number; onChange: (v: number) => void; tone: "liked" | "disliked" }) {
  return (
    <div className="flex">
      {[1, 2, 3, 4, 5].map(s => (
        <button key={s} onClick={() => onChange(s)} aria-label={`${s} stars`} className="p-[1px] transition-transform hover:scale-125">
          <Star className={cn("h-[15px] w-[15px]", s <= value ? (tone === "liked" ? "fill-amber-400 text-amber-400" : "fill-neutral-400 text-neutral-400") : "text-neutral-700")} />
        </button>
      ))}
    </div>
  );
}

export function StarRow({ value, size = 13, className }: { value: number; size?: number; className?: string }) {
  return (
    <span className={cn("inline-flex", className)}>
      {[1, 2, 3, 4, 5].map(s => (
        <Star key={s} style={{ width: size, height: size }} className={s <= Math.round(value) ? "fill-amber-400 text-amber-400" : "text-neutral-700"} />
      ))}
    </span>
  );
}
