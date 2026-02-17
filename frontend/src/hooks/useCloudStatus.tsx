/**
 * Shared cloud status context and hook.
 *
 * Provides a single source of truth for cloud connection status across all components.
 * Uses a React context to share state, so when one component refreshes the status,
 * all components see the updated value immediately.
 *
 * Supports two cloud modes:
 * - Direct mode (VITE_CLOUD_WS_URL): Frontend connects directly to cloud via WebSocket.
 *   Status comes from CloudContext instead of polling.
 * - Proxy mode: Frontend talks to local backend which proxies to cloud.
 *   Status is polled from /api/v1/cloud/status.
 *
 * Polling (proxy mode only) is smart and based on current status:
 * - Disconnected: no polling (state changes only via explicit user actions)
 * - Connecting: poll every 1s to quickly detect when connection completes
 * - Connected: poll every 5s to detect unexpected disconnection
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import { useCloudContext } from "../lib/directCloudContext";

export interface CloudStatus {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  app_id: string | null;
  connection_id: string | null;
  credentials_configured: boolean;
  last_close_code: number | null;
  last_close_reason: string | null;
  /** Whether this status is from direct cloud mode (VITE_CLOUD_WS_URL) */
  is_direct_mode?: boolean;
}

const DEFAULT_STATUS: CloudStatus = {
  connected: false,
  connecting: false,
  error: null,
  app_id: null,
  connection_id: null,
  credentials_configured: false,
  last_close_code: null,
  last_close_reason: null,
  is_direct_mode: false,
};

// Polling intervals based on connection state
const CONNECTING_POLL_INTERVAL = 1000; // 1s - fast polling while waiting for connection
const CONNECTED_POLL_INTERVAL = 5000; // 5s - slow polling to detect unexpected disconnection

interface CloudStatusContextValue {
  status: CloudStatus;
  isLoading: boolean;
  refresh: () => Promise<void>;
}

const CloudStatusContext = createContext<CloudStatusContextValue | null>(null);

interface CloudStatusProviderProps {
  children: ReactNode;
  /** Skip backend polling - used in direct cloud mode where there's no local backend */
  skipPolling?: boolean;
}

/**
 * Provider component that manages cloud status and shares state across all consumers.
 * Polling is automatic and adapts based on connection state.
 */
export function CloudStatusProvider({
  children,
  skipPolling = false,
}: CloudStatusProviderProps) {
  const [status, setStatus] = useState<CloudStatus>(DEFAULT_STATUS);
  const [isLoading, setIsLoading] = useState(!skipPolling);
  const intervalRef = useRef<number | null>(null);

  const fetchStatus = useCallback(async () => {
    // Skip fetching in direct cloud mode - no local backend to poll
    if (skipPolling) {
      return;
    }
    try {
      const response = await fetch("/api/v1/cloud/status");
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch (e) {
      console.error("[CloudStatusProvider] Failed to fetch status:", e);
    } finally {
      setIsLoading(false);
    }
  }, [skipPolling]);

  // Initial fetch on mount (skip in direct cloud mode)
  useEffect(() => {
    if (!skipPolling) {
      fetchStatus();
    }
  }, [fetchStatus, skipPolling]);

  // Smart polling based on connection state (skip in direct cloud mode)
  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Skip polling entirely in direct cloud mode
    if (skipPolling) {
      return;
    }

    // Determine polling interval based on status
    let pollInterval: number | null = null;

    if (status.connecting) {
      // Fast polling while connecting to detect when connection completes
      pollInterval = CONNECTING_POLL_INTERVAL;
    } else if (status.connected) {
      // Slow polling while connected to detect unexpected disconnection
      pollInterval = CONNECTED_POLL_INTERVAL;
    }
    // When disconnected: no polling - state only changes via explicit actions

    if (pollInterval) {
      intervalRef.current = window.setInterval(fetchStatus, pollInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [status.connecting, status.connected, fetchStatus, skipPolling]);

  // Manual refresh function - updates shared state immediately
  const refresh = useCallback(async () => {
    await fetchStatus();
  }, [fetchStatus]);

  return (
    <CloudStatusContext.Provider value={{ status, isLoading, refresh }}>
      {children}
    </CloudStatusContext.Provider>
  );
}

/**
 * Hook to access the shared cloud status.
 *
 * All components using this hook share the same status state.
 * When one component calls refresh(), all components see the updated value immediately.
 *
 * Automatically uses direct cloud mode status when VITE_CLOUD_WS_URL is set,
 * otherwise falls back to proxy mode status from the backend.
 *
 * Must be used within a CloudStatusProvider.
 */
export function useCloudStatus() {
  const context = useContext(CloudStatusContext);

  if (!context) {
    throw new Error("useCloudStatus must be used within a CloudStatusProvider");
  }

  const { status: proxyStatus, isLoading, refresh } = context;

  // Check for direct cloud mode (VITE_CLOUD_WS_URL)
  // This context is optional - if CloudProvider is not in the tree, we just use proxy status
  let cloudContext: ReturnType<typeof useCloudContext> | null = null;
  try {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    cloudContext = useCloudContext();
  } catch {
    // CloudProvider not in tree, use proxy status
  }

  // If in direct cloud mode, use that status instead of proxy status
  const isDirectMode = cloudContext?.isDirectCloudMode ?? false;

  const status: CloudStatus = isDirectMode
    ? {
        connected: cloudContext?.isReady ?? false,
        connecting: cloudContext?.isConnecting ?? false,
        error: cloudContext?.error?.message ?? null,
        app_id: null, // Not available in direct mode
        connection_id: cloudContext?.connectionId ?? null,
        credentials_configured: true, // Credentials are in env vars for direct mode
        last_close_code: cloudContext?.lastCloseCode ?? null,
        last_close_reason: cloudContext?.lastCloseReason ?? null,
        is_direct_mode: true,
      }
    : proxyStatus;

  return {
    status,
    isLoading: isDirectMode ? false : isLoading,
    refresh: isDirectMode ? async () => {} : refresh, // No-op refresh in direct mode
    // Convenience accessors
    isConnected: status.connected,
    isConnecting: status.connecting,
    connectionId: status.connection_id,
    error: status.error,
    lastCloseCode: status.last_close_code,
    lastCloseReason: status.last_close_reason,
    isDirectMode,
  };
}
