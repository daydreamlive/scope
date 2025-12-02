import { defineConfig } from 'vite';
import path from 'path';

// https://vitejs.dev/config
export default defineConfig({
  resolve: {
    browserField: false,
    mainFields: ['module', 'jsnext:main', 'jsnext'],
  },
  build: {
    outDir: path.resolve(__dirname, '.vite/build'),
    lib: {
      entry: path.resolve(__dirname, 'src/preload.ts'),
      formats: ['cjs'],
      fileName: () => 'preload.js',
    },
    rollupOptions: {
      external: [
        'electron',
      ],
      output: {
        entryFileNames: '[name].js',
      },
    },
    emptyOutDir: false, // Don't empty, main.ts already did
    minify: false,
    sourcemap: true,
  },
});
