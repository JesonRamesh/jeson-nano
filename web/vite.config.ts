import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: Vite on :5173 proxies to the Python server on :8000.
// Demo: FastAPI serves the built web/dist directly (no Vite).
const SERVER = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws': { target: SERVER, ws: true },
      '/api': SERVER,
      '/maps': SERVER,
    },
  },
})
