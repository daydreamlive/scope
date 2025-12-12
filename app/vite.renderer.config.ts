import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { copyFileSync, mkdirSync } from 'fs';

// https://vitejs.dev/config
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'copy-log-viewer',
      closeBundle() {
        // Copy LogViewer.html and LogViewer.js to build directory after build
        const files = [
          { src: 'src/components/LogViewer.html', dest: '.vite/build/renderer/LogViewer.html' },
          { src: 'src/components/LogViewer.js', dest: '.vite/build/renderer/LogViewer.js' },
        ];

        files.forEach(({ src, dest }) => {
          const srcPath = path.resolve(__dirname, src);
          const destPath = path.resolve(__dirname, dest);
          try {
            mkdirSync(path.dirname(destPath), { recursive: true });
            copyFileSync(srcPath, destPath);
            console.log(`✓ Copied ${src} to build directory`);
          } catch (err) {
            console.warn(`Failed to copy ${src}:`, err);
          }
        });
      },
    },
  ],
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
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
});
