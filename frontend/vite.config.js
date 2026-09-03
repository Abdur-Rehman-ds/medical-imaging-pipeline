import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-server proxy: forward /v1/* API calls to the FastAPI backend
// (FR-5.x endpoints) running on localhost:8000, avoiding CORS issues.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://localhost:8000',
    },
  },
})
