/**
 * Shared hook for cloud status polling.
 *
 * Centralizes cloud status fetching to avoid redundant polling
 * across multiple components (Header, DaydreamAccountSection, etc.)
 */

import { useState, useEffect, useCallback, useRef } from "react";

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

interface UseCloudStatusOptions {
  /** Polling interval in milliseconds. Default: 5000 */
  pollInterval?: number;
  /** Whether to enable polling. Default: true */
  enabled?: boolean;
}

export function useCloudStatus(options: UseCloudStatusOptions = {}) {
  const { pollInterval = 5000, enabled = true } = options;

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
      console.error("[useCloudStatus] Failed to fetch status:", e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial fetch and polling setup
  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchStatus();

    // Set up polling
    intervalRef.current = window.setInterval(fetchStatus, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, pollInterval, fetchStatus]);

  // Manual refresh function
  const refresh = useCallback(async () => {
    await fetchStatus();
  }, [fetchStatus]);

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
