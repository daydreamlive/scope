import { useEffect, useState } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { PipelinesProvider } from "./contexts/PipelinesContext";
import { CloudProvider } from "./lib/cloudContext";
import { handleOAuthCallback, initElectronAuthListener } from "./lib/auth";
import { toast } from "sonner";
import "./index.css";

// Get cloud WebSocket URL and API key from environment variables
// Set VITE_CLOUD_WS_URL to enable cloud mode, e.g.:
// VITE_CLOUD_WS_URL=wss://fal.run/your-username/scope-app/ws
// VITE_CLOUD_KEY=your-cloud-api-key
const CLOUD_WS_URL = import.meta.env.VITE_CLOUD_WS_URL as string | undefined;
const CLOUD_KEY = import.meta.env.VITE_CLOUD_KEY as string | undefined;

function App() {
  const [isHandlingAuth, setIsHandlingAuth] = useState(true);

  useEffect(() => {
    // Initialize Electron auth callback listener (if running in Electron)
    const cleanupElectronAuth = initElectronAuthListener(
      () => {
        // Success callback - show toast and open account settings
        toast.success("Successfully signed in!");
        window.dispatchEvent(new CustomEvent("daydream-auth-success"));
      },
      error => {
        // Error callback - show toast
        console.error("Electron auth callback error:", error);
        toast.error(`Failed to sign in: ${error.message}`);
      }
    );

    // Handle OAuth callback on mount (for browser flow)
    handleOAuthCallback()
      .then(handled => {
        if (handled) {
          toast.success("Successfully signed in!");
          window.dispatchEvent(new CustomEvent("daydream-auth-success"));
        }
      })
      .catch(error => {
        console.error("OAuth callback error:", error);
        toast.error("Failed to sign in. Please try again.");
      })
      .finally(() => {
        setIsHandlingAuth(false);
      });

    return () => {
      // Cleanup Electron auth listener
      cleanupElectronAuth?.();
    };
  }, []);

  if (isHandlingAuth) {
    // Show a loading state while handling the OAuth callback
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-sm text-muted-foreground">Signing in...</p>
        </div>
      </div>
    );
  }

  return (
    <PipelinesProvider>
      <CloudProvider wsUrl={CLOUD_WS_URL} apiKey={CLOUD_KEY}>
        <StreamPage />
        <Toaster />
      </CloudProvider>
    </PipelinesProvider>
  );
}

export default App;
