import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Vite config specifically for Electron builds
// This ensures the base path is correct for Electron's file:// protocol
export default defineConfig({
  plugins: [react()],
  base: "./", // Use relative paths for Electron
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "../frontend/src"),
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../frontend/dist"),
    emptyOutDir: true,
    // Ensure assets use relative paths
    assetsDir: "assets",
    rollupOptions: {
      output: {
        // Use relative paths for all assets
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
  // No proxy needed in Electron - we'll connect directly to localhost:8000
});
