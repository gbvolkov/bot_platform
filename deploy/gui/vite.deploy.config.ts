import { defineConfig } from 'vite'
import base from './vite.config'

export default defineConfig({
  ...base,
  build: { outDir: 'dist-web', emptyOutDir: true, sourcemap: false },
})
