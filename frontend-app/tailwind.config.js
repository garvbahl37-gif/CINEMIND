import animate from "tailwindcss-animate"

/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{ts,tsx,js,jsx}"
    ],
    theme: {
        extend: {
            colors: {
                // The frame: the electric ultramarine the whole app sits inside.
                ultra: { DEFAULT: "#2B1FBB", hi: "#4636F0", deep: "#170E78" },
                // Second action colour — adding a film to your blend.
                violet: { DEFAULT: "#8A16D4", hi: "#A23BEA" },
                // Only the match stage glows red.
                blood: { DEFAULT: "#BB0001", deep: "#4A0000" },
                ink: "#000000",
                panel: { DEFAULT: "#161616", hi: "#202020" },
                mute: "#9B9B9B",
                paper: "#F4E6D1",

                // shadcn-style tokens, still read by the About page.
                border: "hsl(var(--border))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: "hsl(var(--primary))",
                    foreground: "hsl(var(--primary-foreground))",
                },
                muted: {
                    DEFAULT: "hsl(var(--muted))",
                    foreground: "hsl(var(--muted-foreground))",
                },
            },
            borderRadius: {
                lg: "var(--radius)",
                md: "calc(var(--radius) - 2px)",
                sm: "calc(var(--radius) - 4px)",
            },
            fontFamily: {
                sans: ['"Hanken Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
                display: ['"Hanken Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            },
            keyframes: {
                shimmer: {
                    "0%": { backgroundPosition: "-200% 0" },
                    "100%": { backgroundPosition: "200% 0" },
                },
            },
            animation: {
                shimmer: "shimmer 1.8s linear infinite",
            },
        },
    },
    plugins: [animate],
}
