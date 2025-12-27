import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,     // keep yours
        ws: false,         // no websockets for this route
        timeout: 1200000,   // 120s browser->Vite
        proxyTimeout: 1200000, // 120s Vite->backend
      },
    },
  },
})
