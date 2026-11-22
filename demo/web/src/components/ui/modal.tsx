import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";

// Aceternity-style animated modal: blurred backdrop and a card that springs in.
export function Modal({ open, onClose, title, sub, children }: { open: boolean; onClose: () => void; title: string; sub?: string; children: ReactNode }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc); return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <div className="absolute inset-0 bg-black/70 backdrop-blur-md" onClick={onClose} />
          <motion.div initial={{ opacity: 0, scale: 0.94, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.96, y: 10 }}
            transition={{ type: "spring", bounce: 0.2, duration: 0.5 }}
            className="relative max-h-[90vh] w-full max-w-6xl overflow-auto rounded-3xl border border-white/10 bg-neutral-950 p-7 shadow-2xl">
            <button onClick={onClose} className="absolute right-5 top-5 grid h-9 w-9 place-items-center rounded-full border border-white/10 text-neutral-400 hover:text-white"><X className="h-4 w-4" /></button>
            <h3 className="pr-12 text-2xl font-semibold tracking-tight text-white">{title}</h3>
            {sub && <p className="mt-1 pr-12 text-sm text-neutral-400">{sub}</p>}
            <div className="mt-6">{children}</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
