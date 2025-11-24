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
      "/reserve": {
        // Proxy /reserve to port 8080 (reserve server)
        target: getEnv("SCOPE_RESERVE_URL") || "http://localhost:8080",
        changeOrigin: true,
        secure: getEnv("SCOPE_RESERVE_URL")?.startsWith("https") || false,
        // Configure for streaming/SSE - no timeout, no buffering
        proxyTimeout: 0, // No timeout for streaming connections
        timeout: 0, // No timeout
      },
      "/ping": {
        // Proxy /ping to port 8080 (reserve server) - but also available on main server
        target: getEnv("SCOPE_RESERVE_URL") || "http://localhost:8080",
        changeOrigin: true,
        secure: getEnv("SCOPE_RESERVE_URL")?.startsWith("https") || false,
      },
    },
  },
});
