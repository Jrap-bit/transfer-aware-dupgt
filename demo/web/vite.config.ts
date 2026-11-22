import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built files are served by demo/server.py; during development /api is proxied to it.
export default defineConfig({
  plugins: [react()],
  base: "/",
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
  build: { outDir: "dist", emptyOutDir: true },
});
