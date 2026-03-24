import { useEffect, useState, useRef } from "react";
import { Loader2, CheckCircle2 } from "lucide-react";
import { useCloudStatus } from "../../hooks/useCloudStatus";
import { getDaydreamUserId } from "../../lib/auth";

const ROTATING_MESSAGES = [
  "Establishing connection...",
  "This may take up to 2 minutes",
];
const ROTATE_INTERVAL_MS = 4_000;

/** Fire-and-forget: tell the backend to connect to the cloud relay. */
async function activateCloudRelay() {
  const userId = getDaydreamUserId();
  if (!userId) return;
  try {
    await fetch("/api/v1/cloud/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
  } catch (err) {
    console.error("[Onboarding] Failed to auto-connect to cloud:", err);
  }
}

interface CloudConnectingStepProps {
  onConnected: () => void;
}

export function CloudConnectingStep({ onConnected }: CloudConnectingStepProps) {
  const { isConnected, isConnecting, connectStage, refresh } = useCloudStatus();
  const [msgIndex, setMsgIndex] = useState(0);
  const didConnect = useRef(false);

  // Ensure cloud relay is connecting on mount
  useEffect(() => {
    if (didConnect.current) return;
    didConnect.current = true;
    activateCloudRelay().then(() => refresh());
  }, [refresh]);

  // Keep polling while this step is visible.
  // The global CloudStatusProvider only polls when status.connecting is true,
  // but if we catch the backend before it transitions to "connecting", polling
  // never starts. Poll independently here to guarantee we detect connection.
  useEffect(() => {
    if (isConnected) return;
    const timer = setInterval(refresh, 1_500);
    return () => clearInterval(timer);
  }, [isConnected, refresh]);

  // Rotate messages while waiting
  useEffect(() => {
    if (isConnected) return;
    const timer = setInterval(
      () => setMsgIndex((i) => (i + 1) % ROTATING_MESSAGES.length),
      ROTATE_INTERVAL_MS
    );
    return () => clearInterval(timer);
  }, [isConnected]);

  useEffect(() => {
    if (isConnected) {
      const timer = setTimeout(onConnected, 1_000);
      return () => clearTimeout(timer);
    }
  }, [isConnected, onConnected]);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto text-center">
      <h2 className="text-2xl font-semibold text-foreground">
        Connecting to Daydream Cloud
      </h2>

      {isConnected ? (
        <>
          <CheckCircle2 className="h-8 w-8 text-green-500" />
          <p className="text-sm text-foreground">Connected</p>
        </>
      ) : (
        <>
          <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
          <p className="text-sm text-muted-foreground animate-in fade-in-0 duration-500" key={`${connectStage}-${msgIndex}`}>
            {msgIndex === 0
              ? (isConnecting && connectStage ? connectStage : "Establishing connection...")
              : "This may take up to 2 minutes"}
          </p>
        </>
      )}
    </div>
  );
}
