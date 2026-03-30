import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import {
  initTelemetry,
  getTelemetryEnabled,
  setTelemetryEnabled,
  isEnvTelemetryDisabled,
  isDisclosed as checkDisclosed,
  markDisclosed as persistDisclosed,
  flushQueue as flushTelemetryQueue,
  dropQueue as dropTelemetryQueue,
} from "../lib/telemetry";
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

  useEffect(() => {
    initTelemetry();

    // For existing users who completed onboarding before analytics was added,
    // silently mark disclosure as done and keep telemetry off (opt-in default).
    if (!checkDisclosed()) {
      fetchOnboardingStatus().then(status => {
        if (status.completed) {
          persistDisclosed();
          setIsDisclosed(true);
          dropTelemetryQueue();
        }
      });
    }
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
