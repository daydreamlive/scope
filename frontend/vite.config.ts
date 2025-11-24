import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Helper to get env var (Vite config runs in Node.js context)
declare const process: { env: Record<string, string | undefined> };
const getEnv = (key: string): string | undefined => {
  return process.env[key];
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src",
    },
  },
  server: {
    proxy: {
      "/api": {
        // Default: proxy to localhost:8000
        // Override with SCOPE_API_URL environment variable
        target: getEnv("SCOPE_API_URL") || "http://localhost:8000",
        changeOrigin: true,
        secure: getEnv("SCOPE_API_URL")?.startsWith("https") || false,
        // Vite proxy automatically forwards all headers including Authorization
      },
      "/health": {
        target: getEnv("SCOPE_API_URL") || "http://localhost:8000",
        changeOrigin: true,
        secure: getEnv("SCOPE_API_URL")?.startsWith("https") || false,
      },
    },
  },
});
