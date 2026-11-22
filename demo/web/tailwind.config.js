/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Geist Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"Geist Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        book: { DEFAULT: "#f59e0b", soft: "#fbbf24" },
        movie: { DEFAULT: "#8b5cf6", soft: "#a78bfa" },
      },
      keyframes: {
        spotlight: { "0%": { opacity: 0, transform: "translate(-72%, -62%) scale(0.5)" }, "100%": { opacity: 1, transform: "translate(-50%,-40%) scale(1)" } },
        shimmer: { from: { backgroundPosition: "0 0" }, to: { backgroundPosition: "-200% 0" } },
      },
      animation: {
        spotlight: "spotlight 2s ease .75s 1 forwards",
        shimmer: "shimmer 2.5s linear infinite",
      },
    },
  },
  plugins: [],
};
