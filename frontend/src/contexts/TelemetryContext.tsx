import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import { track } from "../lib/telemetry";
import {
  initTelemetry,
  identifyUser,
  getTelemetryEnabled,
  setTelemetryEnabled,
  isEnvTelemetryDisabled,
  isDisclosed as checkDisclosed,
  markDisclosed as persistDisclosed,
  flushQueue as flushTelemetryQueue,
  dropQueue as dropTelemetryQueue,
} from "../lib/telemetry";
import {
  getDaydreamUserId,
  getDaydreamUserDisplayName,
  getDaydreamUserEmail,
} from "../lib/auth";
import { fetchOnboardingStatus } from "../lib/onboardingStorage";

// ---------------------------------------------------------------------------
// Context shape
// ---------------------------------------------------------------------------

export interface TelemetryContextValue {
  /** Whether telemetry is currently enabled (considers env vars + UI setting) */
  isEnabled: boolean;
  /** Whether an environment variable has forced telemetry off */
  isEnvDisabled: boolean;
  /** Whether the telemetry disclosure has been shown to the user */
  isDisclosed: boolean;
  /** Toggle telemetry on/off and persist the preference */
  setEnabled: (enabled: boolean) => void;
  /** Mark the disclosure as shown (persists to localStorage) */
  markDisclosed: () => void;
  /** Flush pre-disclosure event queue to Mixpanel */
  flushQueue: () => void;
  /** Drop pre-disclosure event queue without sending */
  dropQueue: () => void;
}

const TelemetryContext = createContext<TelemetryContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function TelemetryProvider({ children }: { children: ReactNode }) {
  const [isEnabled, setIsEnabled] = useState(() => getTelemetryEnabled());
  const [isEnvDisabled] = useState(() => isEnvTelemetryDisabled());
  const [isDisclosed, setIsDisclosed] = useState(() => checkDisclosed());

  // Initialize analytics SDK on mount, and if user is already logged in,
  // identify them so events are associated with their daydream account.
  useEffect(() => {
    initTelemetry();
    const userId = getDaydreamUserId();
    if (userId) {
      identifyUser(userId, getDaydreamUserDisplayName(), getDaydreamUserEmail());
    }

    // For existing users who completed onboarding before analytics was added,
    // show an opt-in toast (the onboarding flow handles its own disclosure).
    if (!checkDisclosed()) {
      fetchOnboardingStatus().then((status) => {
        if (status.completed) {
          track("telemetry_disclosure_shown", { path: "existing_user_banner" });
          const shownAt = Date.now();
          toast(
            "Help us improve Scope by sending anonymous usage data. We track UI interactions and feature usage patterns, and we do not collect prompts, parameters, file paths, videos, images, or session replays.",
            {
              duration: Infinity,
              dismissible: false,
              action: {
                label: "Yes",
                onClick: () => {
                  persistDisclosed();
                  setIsDisclosed(true);
                  setTelemetryEnabled(true);
                  setIsEnabled(true);
                  flushTelemetryQueue();
                  track("telemetry_disclosure_responded", {
                    action: "accepted",
                    path: "existing_user_banner",
                    time_to_respond_ms: Date.now() - shownAt,
                    auto_advanced: false,
                  });
                },
              },
              cancel: {
                label: "No thanks",
                onClick: () => {
                  persistDisclosed();
                  setIsDisclosed(true);
                  dropTelemetryQueue();
                  track("telemetry_disclosure_responded", {
                    action: "disabled",
                    path: "existing_user_banner",
                    time_to_respond_ms: Date.now() - shownAt,
                    auto_advanced: false,
                  });
                },
              },
            },
          );
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setEnabled = useCallback((enabled: boolean) => {
    setTelemetryEnabled(enabled);
    setIsEnabled(enabled);
  }, []);

  const markDisclosed = useCallback(() => {
    persistDisclosed();
    setIsDisclosed(true);
  }, []);

  const flushQueue = useCallback(() => {
    flushTelemetryQueue();
  }, []);

  const dropQueue = useCallback(() => {
    dropTelemetryQueue();
  }, []);

  return (
    <TelemetryContext.Provider
      value={{
        isEnabled,
        isEnvDisabled,
        isDisclosed,
        setEnabled,
        markDisclosed,
        flushQueue,
        dropQueue,
      }}
    >
      {children}
    </TelemetryContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useTelemetry(): TelemetryContextValue {
  const ctx = useContext(TelemetryContext);
  if (!ctx) {
    throw new Error("useTelemetry must be used inside <TelemetryProvider>");
  }
  return ctx;
}
