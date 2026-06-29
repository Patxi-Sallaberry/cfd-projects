import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// base relative ('./') -> fonctionne en local ET sur GitHub Pages (sous-chemin /cfd-projects/)
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
})
