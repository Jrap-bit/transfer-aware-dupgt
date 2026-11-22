import { cn } from "../../lib/utils";

export function GridBackground({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute inset-0 [background-size:44px_44px]",
      "[background-image:linear-gradient(to_right,rgba(255,255,255,.045)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,.045)_1px,transparent_1px)]",
      "[mask-image:radial-gradient(ellipse_at_top,black_20%,transparent_70%)]", className)} />
  );
}

export function DotBackground({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute inset-0 [background-size:18px_18px]",
      "[background-image:radial-gradient(rgba(255,255,255,.08)_1px,transparent_1px)]",
      "[mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]", className)} />
  );
}
