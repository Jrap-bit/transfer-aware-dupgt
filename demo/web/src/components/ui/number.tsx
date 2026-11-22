import { animate, useMotionValue, useTransform, motion } from "framer-motion";
import { useEffect } from "react";

// Smoothly animates between numeric values (Magic UI "number ticker").
export function AnimatedNumber({ value, digits = 2, prefix = "", suffix = "", signed = false }:
  { value: number; digits?: number; prefix?: string; suffix?: string; signed?: boolean }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, v => {
    const s = Math.abs(v).toFixed(digits);
    const sign = signed ? (v > 0.0049 ? "+" : v < -0.0049 ? "−" : "±") : v < 0 ? "−" : "";
    return `${prefix}${sign}${s}${suffix}`;
  });
  useEffect(() => { const c = animate(mv, value, { duration: 0.6, ease: [0.22, 1, 0.36, 1] }); return c.stop; }, [value]);
  return <motion.span className="tabular">{text}</motion.span>;
}
