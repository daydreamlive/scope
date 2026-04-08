import { useState, useEffect, useCallback } from "react";
import { useTelemetry } from "../contexts/TelemetryContext";

const GITHUB_TELEMETRY_URL =
  "https://github.com/daydreamlive/scope?tab=readme-ov-file#telemetry";
const AUTO_DISMISS_MS = 30_000;

interface TelemetryBannerProps {
  /** Whether the user has received at least one frame of output. */
  hasReceivedFrames: boolean;
}

export function TelemetryBanner({ hasReceivedFrames }: TelemetryBannerProps) {
  const { isDisclosed, markDisclosed, setEnabled, flushQueue, dropQueue } =
    useTelemetry();
  const [dismissed, setDismissed] = useState(false);

  const handleAccept = useCallback(() => {
    markDisclosed();
    setEnabled(true);
    flushQueue();
    setDismissed(true);
  }, [markDisclosed, setEnabled, flushQueue]);

  const handleDecline = useCallback(() => {
    markDisclosed();
    dropQueue();
    setDismissed(true);
  }, [markDisclosed, dropQueue]);

  // Auto-dismiss after 30s, defaulting to declined
  useEffect(() => {
    if (!hasReceivedFrames || isDisclosed || dismissed) return;
    const timer = setTimeout(() => {
      markDisclosed();
      dropQueue();
      setDismissed(true);
    }, AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [hasReceivedFrames, isDisclosed, dismissed, markDisclosed, dropQueue]);

  if (isDisclosed || dismissed || !hasReceivedFrames) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-between gap-4 bg-background/95 border-t border-border px-4 h-12 text-sm backdrop-blur-sm">
      <span className="text-muted-foreground truncate">
        Help us improve Scope — share anonymous usage data.{" "}
        <a
          href={GITHUB_TELEMETRY_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 hover:text-foreground transition-colors"
          onClick={e => e.stopPropagation()}
        >
          Learn more
        </a>
      </span>
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={handleDecline}
          className="px-3 py-1 text-xs rounded border border-border hover:bg-muted transition-colors"
        >
          No thanks
        </button>
        <button
          onClick={handleAccept}
          className="px-3 py-1 text-xs rounded bg-foreground text-background hover:opacity-80 transition-opacity"
        >
          Yes, share
        </button>
      </div>
    </div>
  );
}
