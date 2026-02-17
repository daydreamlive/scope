/**
 * Cloud Context Provider
 *
 * Provides a context for managing cloud deployment mode.
 * When CLOUD_WS_URL is set, all API calls and WebRTC signaling
 * go through the CloudAdapter WebSocket connection.
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
} from "react";
import { CloudAdapter } from "./directCloudAdapter";

// Check if direct cloud mode is enabled (static, never changes)
const DIRECT_CLOUD_MODE = !!import.meta.env.VITE_CLOUD_WS_URL;

interface CloudContextValue {
  /** Whether we're in direct cloud mode (VITE_CLOUD_WS_URL is set) - STATIC, always true if env var is set */
  isDirectCloudMode: boolean;
  /** The CloudAdapter instance (null if not in cloud mode or not connected) */
  adapter: CloudAdapter | null;
  /** Whether we're currently connecting (waiting for ready message) */
  isConnecting: boolean;
  /** Whether the adapter is connected and ready */
  isReady: boolean;
  /** Connection error if any */
  error: Error | null;
  /** Connection ID assigned by the cloud server */
  connectionId: string | null;
  /** Last close code if connection was closed unexpectedly */
  lastCloseCode: number | null;
  /** Last close reason if connection was closed unexpectedly */
  lastCloseReason: string | null;
  /** Disconnect the cloud adapter (for logout) */
  disconnect: () => void;
  /** Reconnect the cloud adapter (for login) */
  reconnect: () => void;
}

const CloudContext = createContext<CloudContextValue>({
  isDirectCloudMode: DIRECT_CLOUD_MODE,
  adapter: null,
  isConnecting: false,
  isReady: false,
  error: null,
  connectionId: null,
  lastCloseCode: null,
  lastCloseReason: null,
  disconnect: () => {},
  reconnect: () => {},
});

interface CloudProviderProps {
  /** WebSocket URL for cloud endpoint. If not set, local mode is used. */
  wsUrl?: string;
  /** cloud API key for authentication */
  apiKey?: string;
  /** User ID for log correlation (sent to cloud after connection) */
  userId?: string;
  children: React.ReactNode;
}

export function CloudProvider({
  wsUrl,
  apiKey,
  userId,
  children,
}: CloudProviderProps) {
  const [adapter, setAdapter] = useState<CloudAdapter | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [lastCloseCode, setLastCloseCode] = useState<number | null>(null);
  const [lastCloseReason, setLastCloseReason] = useState<string | null>(null);
  // Track if manually disconnected (e.g., user logged out)
  const [manuallyDisconnected, setManuallyDisconnected] = useState(false);
  // Track if the effect is still active to prevent state updates after cleanup
  // This is important for React StrictMode which double-mounts components
  const isActiveRef = useRef(true);
  // Store adapter ref for disconnect/reconnect
  const adapterRef = useRef<CloudAdapter | null>(null);

  useEffect(() => {
    // Reset active flag on each effect run
    isActiveRef.current = true;

    if (!wsUrl || manuallyDisconnected) {
      // Don't connect if no URL or manually disconnected
      if (adapterRef.current) {
        adapterRef.current.disconnect();
        adapterRef.current = null;
      }
      setAdapter(null);
      setIsConnecting(false);
      setIsReady(false);
      setError(null);
      setConnectionId(null);
      setLastCloseCode(null);
      setLastCloseReason(null);
      return;
    }

    console.log("[CloudProvider] Connecting to cloud:", wsUrl);
    const cloudAdapter = new CloudAdapter(wsUrl, apiKey, userId);
    adapterRef.current = cloudAdapter;
    setAdapter(cloudAdapter);
    // Set connecting state while waiting for ready message
    setIsConnecting(true);
    setIsReady(false);
    setError(null);

    cloudAdapter
      .connect()
      .then(() => {
        // Only update state if this effect is still active
        if (isActiveRef.current) {
          console.log("[CloudProvider] Connected to cloud");
          setIsConnecting(false);
          setIsReady(true);
          setError(null);
          setConnectionId(cloudAdapter.connectionId);
          // Clear close info on successful connection
          setLastCloseCode(null);
          setLastCloseReason(null);
        } else {
          // Effect was cleaned up while connecting, disconnect immediately
          console.log(
            "[CloudProvider] Connection completed but effect was cleaned up, disconnecting"
          );
          cloudAdapter.disconnect();
        }
      })
      .catch(err => {
        // Only update state if this effect is still active
        if (isActiveRef.current) {
          console.error("[CloudProvider] Connection failed:", err);
          setIsConnecting(false);
          setError(err);
          setIsReady(false);
          setConnectionId(null);
          // Update close info from adapter
          setLastCloseCode(cloudAdapter.lastCloseCode);
          setLastCloseReason(cloudAdapter.lastCloseReason);
        }
      });

    return () => {
      console.log("[CloudProvider] Disconnecting from cloud");
      // Mark effect as inactive before disconnecting
      isActiveRef.current = false;
      cloudAdapter.disconnect();
      adapterRef.current = null;
    };
  }, [wsUrl, apiKey, userId, manuallyDisconnected]);

  // Disconnect function for logout
  const disconnect = React.useCallback(() => {
    console.log("[CloudProvider] Manual disconnect requested");
    setManuallyDisconnected(true);
  }, []);

  // Reconnect function for login
  const reconnect = React.useCallback(() => {
    console.log("[CloudProvider] Reconnect requested");
    setManuallyDisconnected(false);
  }, []);

  const value: CloudContextValue = {
    isDirectCloudMode: DIRECT_CLOUD_MODE,
    adapter,
    isConnecting: !!wsUrl && !manuallyDisconnected && isConnecting,
    isReady: !!wsUrl && !manuallyDisconnected && isReady,
    error,
    connectionId,
    lastCloseCode,
    lastCloseReason,
    disconnect,
    reconnect,
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
  const {
    adapter,
    isDirectCloudMode,
    isConnecting,
    isReady,
    error,
    connectionId,
    lastCloseCode,
    lastCloseReason,
    disconnect,
    reconnect,
  } = useCloudContext();
  return {
    adapter,
    isDirectCloudMode,
    isConnecting,
    isReady,
    error,
    connectionId,
    lastCloseCode,
    lastCloseReason,
    disconnect,
    reconnect,
  };
}
