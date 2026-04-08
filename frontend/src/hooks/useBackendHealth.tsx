/**
 * Backend health monitoring context and hook.
 *
 * Detects when the Python backend crashes or becomes unresponsive:
 * - In Electron: listens to IPC server-status events for instant crash detection
 * - In all modes: polls /health every 10s, marks backend offline after 4 consecutive failures
 * - Skips failure counting until the first successful health check (startup grace period)
 *
 * Shows a persistent toast when backend goes offline and a recovery toast when it comes back.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { toast } from "sonner";

const HEALTH_POLL_INTERVAL_MS = 10_000;
const FAILURE_THRESHOLD = 4; // consecutive failures before marking offline
const HEALTH_FETCH_TIMEOUT_MS = 5_000;

interface BackendHealthContextValue {
  isHealthy: boolean;
}

const BackendHealthContext = createContext<BackendHealthContextValue>({
  isHealthy: true,
});

interface BackendHealthProviderProps {
  children: ReactNode;
}

export function BackendHealthProvider({ children }: BackendHealthProviderProps) {
  const [isHealthy, setIsHealthy] = useState(true);

  // Use refs so callbacks can read/write state without stale closures
  const isHealthyRef = useRef(true);
  const consecutiveFailuresRef = useRef(0);
  const offlineToastIdRef = useRef<string | number | null>(null);
  // Only show recovery toast if we've ever gone offline during this session
  const hasGoneOfflineRef = useRef(false);
  // Don't count polling failures until the backend has responded at least once
  const hasEverConnectedRef = useRef(false);
  // Ref to the health check function so the toast retry button can trigger it
  const checkHealthNowRef = useRef<(() => void) | null>(null);

  const markHealthy = () => {
    consecutiveFailuresRef.current = 0;
    hasEverConnectedRef.current = true;
    if (!isHealthyRef.current) {
      isHealthyRef.current = true;
      setIsHealthy(true);
    }
  };

  const markUnhealthy = () => {
    if (isHealthyRef.current) {
      isHealthyRef.current = false;
      setIsHealthy(false);
    }
  };

  // Show/dismiss toasts when health state changes
  useEffect(() => {
    if (isHealthy) {
      if (offlineToastIdRef.current !== null) {
        toast.dismiss(offlineToastIdRef.current);
        offlineToastIdRef.current = null;
      }
      if (hasGoneOfflineRef.current) {
        toast.success("Backend reconnected", { duration: 4000 });
      }
    } else {
      hasGoneOfflineRef.current = true;
      console.error("[BackendHealth] Backend is unresponsive or has crashed");
      offlineToastIdRef.current = toast.error("Backend is not responding", {
        description:
          "If this persists, try restarting the app.",
        duration: Infinity,
        action: {
          label: "Retry now",
          onClick: () => checkHealthNowRef.current?.(),
        },
      });
    }
  }, [isHealthy]);

  // Electron IPC: listen for instant server status events (registered once)
  useEffect(() => {
    if (!window.scope?.onServerStatus) return;
    return window.scope.onServerStatus((isRunning: boolean) => {
      if (isRunning) {
        markHealthy();
      } else {
        markUnhealthy();
      }
    });
    // markHealthy/markUnhealthy are defined in render scope but read from refs,
    // so they are stable — no deps needed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Electron IPC: listen for server error events (registered once)
  useEffect(() => {
    if (!window.scope?.onServerError) return;
    return window.scope.onServerError((error: string) => {
      console.error("[BackendHealth] Server error from IPC:", error);
      markUnhealthy();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Health polling: detect unresponsiveness in all modes
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(
          () => controller.abort(),
          HEALTH_FETCH_TIMEOUT_MS
        );
        const resp = await fetch("/health", { signal: controller.signal });
        clearTimeout(timeoutId);

        if (resp.ok) {
          markHealthy();
        } else {
          consecutiveFailuresRef.current++;
          if (
            hasEverConnectedRef.current &&
            consecutiveFailuresRef.current >= FAILURE_THRESHOLD
          ) {
            markUnhealthy();
          }
        }
      } catch {
        consecutiveFailuresRef.current++;
        if (
          hasEverConnectedRef.current &&
          consecutiveFailuresRef.current >= FAILURE_THRESHOLD
        ) {
          markUnhealthy();
        }
      }
    };

    checkHealthNowRef.current = checkHealth;
    checkHealth();
    const interval = setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS);
    return () => {
      clearInterval(interval);
      checkHealthNowRef.current = null;
    };
  }, []);

  return (
    <BackendHealthContext.Provider value={{ isHealthy }}>
      {children}
    </BackendHealthContext.Provider>
  );
}

export function useBackendHealth() {
  return useContext(BackendHealthContext);
}
