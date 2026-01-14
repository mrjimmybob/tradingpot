/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom color palette from design system
        profit: '#22c55e',  // Green for profits
        loss: '#ef4444',    // Red for losses
        running: '#3b82f6', // Blue for running status
        paused: '#eab308',  // Yellow for paused status
        stopped: '#6b7280', // Gray for stopped status
        accent: '#6366f1',  // Indigo accent
      },
      textColor: {
        profit: '#22c55e',
        loss: '#ef4444',
        running: '#3b82f6',
        paused: '#eab308',
        stopped: '#6b7280',
        accent: '#6366f1',
      },
      ringColor: {
        accent: '#6366f1',
        profit: '#22c55e',
        loss: '#ef4444',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Monaco', 'monospace'],
      },
    },
  },
  plugins: [],
}
