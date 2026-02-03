/**
 * Cloud Context Provider
 *
 * Provides a context for managing cloud deployment mode.
 * When CLOUD_WS_URL is set, all API calls and WebRTC signaling
 * go through the CloudAdapter WebSocket connection.
 */

import React, { createContext, useContext, useEffect, useState } from "react";
import { CloudAdapter } from "./cloudAdapter";

interface CloudContextValue {
  /** Whether we're in cloud mode */
  isCloudMode: boolean;
  /** The CloudAdapter instance (null if not in cloud mode) */
  adapter: CloudAdapter | null;
  /** Whether the adapter is connected and ready */
  isReady: boolean;
  /** Connection error if any */
  error: Error | null;
}

const CloudContext = createContext<CloudContextValue>({
  isCloudMode: false,
  adapter: null,
  isReady: false,
  error: null,
});

interface CloudProviderProps {
  /** WebSocket URL for cloud endpoint. If not set, local mode is used. */
  wsUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  children: React.ReactNode;
}

export function CloudProvider({ wsUrl, apiKey, children }: CloudProviderProps) {
  const [adapter, setAdapter] = useState<CloudAdapter | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!wsUrl) {
      setAdapter(null);
      setIsReady(false);
      setError(null);
      return;
    }

    console.log("[CloudProvider] Connecting to cloud:", wsUrl);
    const cloudAdapter = new CloudAdapter(wsUrl, apiKey);
    setAdapter(cloudAdapter);

    cloudAdapter
      .connect()
      .then(() => {
        console.log("[CloudProvider] Connected to cloud");
        setIsReady(true);
        setError(null);
      })
      .catch(err => {
        console.error("[CloudProvider] Connection failed:", err);
        setError(err);
        setIsReady(false);
      });

    return () => {
      console.log("[CloudProvider] Disconnecting from cloud");
      cloudAdapter.disconnect();
    };
  }, [wsUrl]);

  const value: CloudContextValue = {
    isCloudMode: !!wsUrl,
    adapter,
    isReady: !!wsUrl && isReady,
    error,
  };

  return (
    <CloudContext.Provider value={value}>{children}</CloudContext.Provider>
  );
}

/**
 * Hook to access the cloud context
 */
export function useCloudContext() {
  return useContext(CloudContext);
}

/**
 * Hook that returns the adapter if in cloud mode
 */
export function useCloudAdapter() {
  const { adapter, isCloudMode, isReady, error } = useCloudContext();
  return { adapter, isCloudMode, isReady, error };
}
