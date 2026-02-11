/**
 * Shared hook for cloud status polling.
 *
 * Centralizes cloud status fetching to avoid redundant polling
 * across multiple components (Header, DaydreamAccountSection, etc.)
 */

import { useQuery } from "@tanstack/react-query";
import { queryKeys } from "./queries/queryKeys";

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

async function fetchCloudStatus(): Promise<CloudStatus> {
  const response = await fetch("/api/v1/cloud/status");
  if (!response.ok) {
    throw new Error(`Cloud status fetch failed: ${response.status}`);
  }
  return response.json();
}

export function useCloudStatus(options: UseCloudStatusOptions = {}) {
  const { pollInterval = 5000, enabled = true } = options;

  const { data, isLoading, refetch } = useQuery<CloudStatus>({
    queryKey: queryKeys.cloudStatus(),
    queryFn: fetchCloudStatus,
    refetchInterval: enabled ? pollInterval : false,
    enabled,
    placeholderData: DEFAULT_STATUS,
  });

  const status = data ?? DEFAULT_STATUS;

  return {
    status,
    isLoading,
    refresh: refetch,
    // Convenience accessors
    isConnected: status.connected,
    isConnecting: status.connecting,
    connectionId: status.connection_id,
    error: status.error,
    lastCloseCode: status.last_close_code,
    lastCloseReason: status.last_close_reason,
  };
}
