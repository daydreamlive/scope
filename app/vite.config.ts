import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron';
import path from 'path';

// https://vitejs.dev/config
export default defineConfig({
  plugins: [
    react(),
    electron([
      {
        // Main process entry file
        entry: 'src/main.ts',
        vite: {
          build: {
            outDir: '.vite/build',
            rollupOptions: {
              external: [
                'electron',
                'electron-log',
                'electron-updater',
                'fs-extra',
              ],
            },
          },
        },
      },
      {
        // Preload scripts
        entry: 'src/preload.ts',
        vite: {
          build: {
            outDir: '.vite/build',
          },
        },
        onstart(options) {
          // Reload renderer on preload change
          options.reload();
        },
      },
    ]),
  ],
  root: path.resolve(__dirname),
  base: './',
  build: {
    outDir: path.resolve(__dirname, '.vite/build/renderer'),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main_window: path.resolve(__dirname, 'index.html'),
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
});
