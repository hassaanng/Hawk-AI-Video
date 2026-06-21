/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0d0c",
          900: "#101413",
          800: "#161b1a",
          700: "#1f2624",
          600: "#2a3330",
          500: "#3a4540",
        },
        signal: {
          400: "#7df27a",
          500: "#5fe65a",
          600: "#43c93f",
        },
        amber: {
          400: "#f2b84f",
          500: "#e6a432",
        },
        crimson: {
          400: "#f2685f",
          500: "#e8453a",
        },
        cyan: {
          400: "#5fd8e6",
        },
      },
      fontFamily: {
        mono: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.02em" }],
      },
    },
  },
  plugins: [],
};
