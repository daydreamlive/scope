/**
 * Unified API hook that automatically routes requests through CloudAdapter
 * when in cloud mode, or uses direct HTTP when in local mode.
 */

import { useMemo } from "react";
import { useCloudContext } from "../lib/cloudContext";
import { createApiClient } from "../lib/unifiedApi";

/**
 * Hook that provides API functions that work in both local and cloud modes.
 *
 * In cloud mode, all requests go through the CloudAdapter WebSocket.
 * In local mode, requests go directly via HTTP fetch.
 */
export function useApi() {
  const { adapter, isCloudMode, isReady } = useCloudContext();

  const client = useMemo(
    () => createApiClient(adapter, isCloudMode),
    [adapter, isCloudMode]
  );

  return {
    ...client,
    isCloudMode,
    isReady,
  };
}
