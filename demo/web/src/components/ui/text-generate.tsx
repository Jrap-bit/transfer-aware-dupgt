import { motion } from "framer-motion";

// Aceternity "text generate effect": words fade and un-blur in sequence.
export function TextGenerate({ words, className, highlight = [] }: { words: string; className?: string; highlight?: number[] }) {
  return (
    <span className={className}>
      {words.split(" ").map((w, i) => (
        <motion.span key={i + w} initial={{ opacity: 0, filter: "blur(8px)" }} animate={{ opacity: 1, filter: "blur(0px)" }}
          transition={{ duration: 0.5, delay: i * 0.08 }}
          className={highlight.includes(i) ? "bg-gradient-to-r from-amber-300 via-orange-300 to-violet-400 bg-clip-text text-transparent" : ""}>
          {w}{" "}
        </motion.span>
      ))}
    </span>
  );
}
