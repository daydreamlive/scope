import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config
export default defineConfig({
  plugins: [react()],
  root: path.resolve(__dirname),
  base: './',
  build: {
    outDir: path.resolve(__dirname, '.vite/build/renderer'),
    emptyOutDir: false,
    rollupOptions: {
      input: {
        main_window: path.resolve(__dirname, 'index.html'),
      },
    },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
});
