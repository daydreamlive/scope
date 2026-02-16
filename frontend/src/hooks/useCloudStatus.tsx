/**
 * Shared cloud status context and hook.
 *
 * Provides a single source of truth for cloud connection status across all components.
 * Uses a React context to share state, so when one component refreshes the status,
 * all components see the updated value immediately.
 *
 * Polling is smart and based on current status:
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

export interface CloudStatus {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  app_id: string | null;
  connection_id: string | null;
  credentials_configured: boolean;
  last_close_code: number | null;
  last_close_reason: string | null;
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
}

/**
 * Provider component that manages cloud status and shares state across all consumers.
 * Polling is automatic and adapts based on connection state.
 */
export function CloudStatusProvider({ children }: CloudStatusProviderProps) {
  const [status, setStatus] = useState<CloudStatus>(DEFAULT_STATUS);
  const [isLoading, setIsLoading] = useState(true);
  const intervalRef = useRef<number | null>(null);

  const fetchStatus = useCallback(async () => {
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
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Smart polling based on connection state
  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
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
  }, [status.connecting, status.connected, fetchStatus]);

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
 * Must be used within a CloudStatusProvider.
 */
export function useCloudStatus() {
  const context = useContext(CloudStatusContext);

  if (!context) {
    throw new Error("useCloudStatus must be used within a CloudStatusProvider");
  }

  const { status, isLoading, refresh } = context;

  return {
    status,
    isLoading,
    refresh,
    // Convenience accessors
    isConnected: status.connected,
    isConnecting: status.connecting,
    connectionId: status.connection_id,
    error: status.error,
    lastCloseCode: status.last_close_code,
    lastCloseReason: status.last_close_reason,
  };
}
