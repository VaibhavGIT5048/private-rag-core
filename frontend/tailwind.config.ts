import type { Config } from 'tailwindcss'

// Colour/spacing tokens live in globals.css as CSS custom properties so the
// Light / Dark theme switch is a single attribute flip at runtime. Tailwind
// is used for layout; the vars carry the identity.
export default {
  content: ['./app/**/*.{ts,tsx}', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        ink: 'var(--ink)',
        accent: 'var(--accent)',
        'accent-hi': 'var(--accent-hi)',
        'on-accent': 'var(--on-accent)',
        panel: 'var(--panel)',
        'panel-solid': 'var(--panel-solid)',
        'panel-hi': 'var(--panel-hi)',
        'chip-bg': 'var(--chip-bg)',
        brd: 'var(--brd)',
      },
      fontFamily: {
        sans: ['var(--font-archivo)', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
