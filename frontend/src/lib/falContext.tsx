/**
 * Fal.ai Context Provider
 *
 * Provides a context for managing fal.ai deployment mode.
 * When FAL_WS_URL is set, all API calls and WebRTC signaling
 * go through the FalAdapter WebSocket connection.
 */

import React, { createContext, useContext, useEffect, useState } from "react";
import { FalAdapter } from "./falAdapter";

interface FalContextValue {
  /** Whether we're in fal mode */
  isFalMode: boolean;
  /** The FalAdapter instance (null if not in fal mode) */
  adapter: FalAdapter | null;
  /** Whether the adapter is connected and ready */
  isReady: boolean;
  /** Connection error if any */
  error: Error | null;
}

const FalContext = createContext<FalContextValue>({
  isFalMode: false,
  adapter: null,
  isReady: false,
  error: null,
});

interface FalProviderProps {
  /** WebSocket URL for fal.ai endpoint. If not set, local mode is used. */
  wsUrl?: string;
  /** fal.ai API key for authentication */
  apiKey?: string;
  children: React.ReactNode;
}

export function FalProvider({ wsUrl, apiKey, children }: FalProviderProps) {
  const [adapter, setAdapter] = useState<FalAdapter | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!wsUrl) {
      setAdapter(null);
      setIsReady(false);
      setError(null);
      return;
    }

    console.log("[FalProvider] Connecting to fal.ai:", wsUrl);
    const falAdapter = new FalAdapter(wsUrl, apiKey);
    setAdapter(falAdapter);

    falAdapter
      .connect()
      .then(() => {
        console.log("[FalProvider] Connected to fal.ai");
        setIsReady(true);
        setError(null);
      })
      .catch((err) => {
        console.error("[FalProvider] Connection failed:", err);
        setError(err);
        setIsReady(false);
      });

    return () => {
      console.log("[FalProvider] Disconnecting from fal.ai");
      falAdapter.disconnect();
    };
  }, [wsUrl]);

  const value: FalContextValue = {
    isFalMode: !!wsUrl,
    adapter,
    isReady: !!wsUrl && isReady,
    error,
  };

  return <FalContext.Provider value={value}>{children}</FalContext.Provider>;
}

/**
 * Hook to access the fal context
 */
export function useFalContext() {
  return useContext(FalContext);
}

/**
 * Hook that returns the adapter if in fal mode
 */
export function useFalAdapter() {
  const { adapter, isFalMode, isReady, error } = useFalContext();
  return { adapter, isFalMode, isReady, error };
}
