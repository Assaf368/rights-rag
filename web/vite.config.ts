import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // In development the API runs on port 8000 (uvicorn rights_rag.api:app).
  server: { proxy: { '/ask': 'http://localhost:8000', '/health': 'http://localhost:8000' } },
})
