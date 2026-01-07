import { useEffect, useState } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { handleOAuthCallback } from "./lib/auth";
import { toast } from "sonner";
import "./index.css";

function App() {
  const [isHandlingAuth, setIsHandlingAuth] = useState(true);

  useEffect(() => {
    // Handle OAuth callback on mount
    handleOAuthCallback()
      .then((handled) => {
        if (handled) {
          toast.success("Successfully signed in!");
        }
      })
      .catch((error) => {
        console.error("OAuth callback error:", error);
        toast.error("Failed to sign in. Please try again.");
      })
      .finally(() => {
        setIsHandlingAuth(false);
      });
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
    <>
      <StreamPage />
      {/* <Toaster /> */}
    </>
  );
}

export default App;
