import { useEffect, useRef, useState, useCallback } from "react";
import { BarChart3 } from "lucide-react";
import { trackEvent } from "../../lib/analytics";

interface TelemetryDisclosureProps {
  onAccept: () => void;
  onDecline: () => void;
  /** If set, auto-advances (accepts) after this many seconds. */
  autoAdvanceSeconds?: number;
  /** Disclosure path for tracking. */
  path: "cloud_wait" | "local_interstitial" | "existing_user_banner";
}

export function TelemetryDisclosure({
  onAccept,
  onDecline,
  autoAdvanceSeconds,
  path,
}: TelemetryDisclosureProps) {
  const [secondsLeft, setSecondsLeft] = useState(autoAdvanceSeconds ?? 0);
  const shownTimeRef = useRef(Date.now());
  const trackedShown = useRef(false);

  // Track disclosure shown once
  useEffect(() => {
    if (trackedShown.current) return;
    trackedShown.current = true;
    trackEvent("telemetry_disclosure_shown", { path });
  }, [path]);

  // Auto-advance countdown
  useEffect(() => {
    if (!autoAdvanceSeconds || autoAdvanceSeconds <= 0) return;
    const timer = setInterval(() => {
      setSecondsLeft(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [autoAdvanceSeconds]);

  // Fire auto-advance when countdown hits 0 (defaults to decline for opt-in)
  useEffect(() => {
    if (autoAdvanceSeconds && secondsLeft === 0) {
      trackEvent("telemetry_disclosure_responded", {
        action: "declined",
        path,
        time_to_respond_ms: Date.now() - shownTimeRef.current,
        auto_advanced: true,
      });
      onDecline();
    }
    // Only run when secondsLeft changes to 0
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [secondsLeft]);

  const handleAccept = useCallback(() => {
    trackEvent("telemetry_disclosure_responded", {
      action: "accepted",
      path,
      time_to_respond_ms: Date.now() - shownTimeRef.current,
      auto_advanced: false,
    });
    onAccept();
  }, [onAccept, path]);

  const handleDecline = useCallback(() => {
    trackEvent("telemetry_disclosure_responded", {
      action: "disabled",
      path,
      time_to_respond_ms: Date.now() - shownTimeRef.current,
      auto_advanced: false,
    });
    onDecline();
  }, [onDecline, path]);

  const progressPct = autoAdvanceSeconds
    ? ((autoAdvanceSeconds - secondsLeft) / autoAdvanceSeconds) * 100
    : 0;

  return (
    <div className="w-full max-w-md mx-auto animate-in fade-in-0 slide-in-from-bottom-4 duration-500">
      <div className="rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-muted">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </div>
          <h3 className="text-sm font-medium text-foreground">
            Usage Analytics
          </h3>
        </div>

        <p className="text-sm text-muted-foreground leading-relaxed">
          Help us improve Scope by sending anonymous usage data. We track UI
          interactions and feature usage patterns, and we do not collect
          prompts, parameters, file paths, videos, images, or session replays.
        </p>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDecline}
            className="flex-1 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted/50 transition-colors text-foreground"
          >
            No thanks
          </button>
          <button
            onClick={handleAccept}
            className="flex-1 px-4 py-2 text-sm font-medium rounded-lg bg-foreground text-background hover:bg-foreground/90 transition-colors"
          >
            Yes
          </button>
        </div>

        <a
          href="https://github.com/daydreamlive/scope/tree/main/docs/telemetry.md"
          target="_blank"
          rel="noopener noreferrer"
          className="block text-xs text-muted-foreground hover:text-foreground transition-colors text-center"
        >
          Learn more about our approach
        </a>

        {/* Auto-advance progress bar */}
        {autoAdvanceSeconds && secondsLeft > 0 && (
          <div className="h-0.5 w-full bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-muted-foreground/40 transition-all duration-1000 ease-linear"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
