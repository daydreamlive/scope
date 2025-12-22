import { defineConfig } from 'vite';
import path from 'path';

// https://vitejs.dev/config
export default defineConfig({
  resolve: {
    // Electron's main process runs in Node.js, so we need Node.js-compatible modules
    browserField: false,
    mainFields: ['module', 'jsnext:main', 'jsnext'],
  },
  build: {
    outDir: path.resolve(__dirname, '.vite/build'),
    lib: {
      entry: path.resolve(__dirname, 'src/main.ts'),
      formats: ['cjs'],
      fileName: () => 'main.js',
    },
    rollupOptions: {
      external: [
        'electron',
        'electron-log',
        'electron-updater',
        'path',
        'fs',
        'fs-extra',
        'child_process',
        'os',
        'http',
        'https',
        'url',
        'stream',
        'util',
        'events',
        'assert',
        'buffer',
        'crypto',
        'net',
        'tls',
        'zlib',
      ],
      output: {
        entryFileNames: '[name].js',
      },
    },
    emptyOutDir: true,
    minify: false,
    sourcemap: true,
  },
});
