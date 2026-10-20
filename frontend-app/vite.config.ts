import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// In development, /api is proxied to a running API so the app stays
// same-origin, as it is in production. Point it at a local uvicorn with
// API_PROXY=http://localhost:8000; it defaults to the deployed API.
const apiTarget = process.env.API_PROXY ?? 'https://cinemind-theta.vercel.app'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true, secure: true },
    },
  },
})
