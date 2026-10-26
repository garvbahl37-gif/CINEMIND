import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// In development, /api is proxied to a running API so the app stays
// same-origin, as it is in production. Point it at a local uvicorn with
// API_PROXY=http://localhost:8000; it defaults to the deployed API.
const apiTarget = process.env.API_PROXY ?? 'https://cinemind-theta.vercel.app'

// Each build gets an id that its API reads carry, so after a deploy a browser asks for
// new URLs instead of showing the last deploy's cached answers (they are cached with
// stale-while-revalidate for a week).
const buildId = (process.env.VERCEL_GIT_COMMIT_SHA ?? '').slice(0, 12) || Date.now().toString(36)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: { __BUILD_ID__: JSON.stringify(buildId) },
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
