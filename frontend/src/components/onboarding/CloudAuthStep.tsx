import { useState, useEffect, useRef } from "react";
import { Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "../ui/button";
import {
  isAuthenticated,
  getDaydreamUserDisplayName,
  getDaydreamUserId,
  redirectToSignIn,
} from "../../lib/auth";
import { activateCloudRelay } from "../../lib/onboardingStorage";

type AuthState = "idle" | "waiting" | "success" | "error";

const AUTH_TIMEOUT_MS = 60_000;
const AUTO_ADVANCE_DELAY_MS = 2_000;

interface CloudAuthStepProps {
  onComplete: () => void;
}

export function CloudAuthStep({ onComplete }: CloudAuthStepProps) {
  const [authState, setAuthState] = useState<AuthState>(() =>
    isAuthenticated() ? "success" : "idle"
  );
  const [displayName, setDisplayName] = useState<string | null>(() =>
    getDaydreamUserDisplayName()
  );
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const advanceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-advance when already authenticated on mount
  useEffect(() => {
    if (authState === "success") {
      setDisplayName(getDaydreamUserDisplayName());
      activateCloudRelay(getDaydreamUserId());
      advanceRef.current = setTimeout(onComplete, AUTO_ADVANCE_DELAY_MS);
    }
    return () => {
      if (advanceRef.current) clearTimeout(advanceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Listen for auth events dispatched by App.tsx
  useEffect(() => {
    const handleSuccess = () => {
      setAuthState("success");
      setDisplayName(getDaydreamUserDisplayName());
      activateCloudRelay(getDaydreamUserId());
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      advanceRef.current = setTimeout(onComplete, AUTO_ADVANCE_DELAY_MS);
    };
    const handleError = () => {
      setAuthState("error");
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };

    window.addEventListener("daydream-auth-success", handleSuccess);
    window.addEventListener("daydream-auth-error", handleError);
    return () => {
      window.removeEventListener("daydream-auth-success", handleSuccess);
      window.removeEventListener("daydream-auth-error", handleError);
    };
  }, [onComplete]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (advanceRef.current) clearTimeout(advanceRef.current);
    };
  }, []);

  const handleSignIn = () => {
    setAuthState("waiting");
    redirectToSignIn();
    // Timeout after 60s
    timeoutRef.current = setTimeout(() => {
      setAuthState("error");
    }, AUTH_TIMEOUT_MS);
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto text-center">
      <h2 className="text-2xl font-semibold text-foreground">
        Sign in to your Daydream account
      </h2>

      {authState === "idle" && (
        <>
          <p className="text-sm text-muted-foreground">
            Sign in to use Daydream Cloud for real-time AI inference without a
            local GPU.
          </p>
          <Button onClick={handleSignIn} size="lg" className="px-8">
            Sign In
          </Button>
        </>
      )}

      {authState === "waiting" && (
        <>
          <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
          <p className="text-sm text-muted-foreground">
            Waiting for sign-in to complete in your browser...
          </p>
        </>
      )}

      {authState === "success" && (
        <>
          <CheckCircle2 className="h-8 w-8 text-green-500" />
          <p className="text-sm text-foreground">
            Signed in as{" "}
            <span className="font-medium">{displayName ?? "you"}</span>
          </p>
        </>
      )}

      {authState === "error" && (
        <>
          <AlertCircle className="h-8 w-8 text-destructive" />
          <p className="text-sm text-muted-foreground">
            Something went wrong. Please try again.
          </p>
          <div className="flex items-center gap-3">
            <Button onClick={handleSignIn} variant="outline">
              Try Again
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
