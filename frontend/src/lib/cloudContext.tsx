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
  useRef,
  useState,
} from "react";
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
  /** Static API key — used as fallback when no tokenProvider is set */
  apiKey?: string;
  /** Dynamic token provider — called to get short-lived inference tokens */
  tokenProvider?: () => Promise<string | null>;
  children: React.ReactNode;
}

export function CloudProvider({
  wsUrl,
  apiKey,
  tokenProvider,
  children,
}: CloudProviderProps) {
  const [adapter, setAdapter] = useState<CloudAdapter | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const tokenRefreshRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const adapterRef = useRef<CloudAdapter | null>(null);

  useEffect(() => {
    if (!wsUrl) {
      setAdapter(null);
      setIsReady(false);
      setError(null);
      return;
    }

    let cancelled = false;

    async function connectWithToken() {
      let token = apiKey;

      // If tokenProvider is available, use it to get a dynamic token
      if (tokenProvider) {
        const dynamicToken = await tokenProvider();
        if (cancelled) return;
        if (!dynamicToken) {
          console.log("[CloudProvider] No inference token — cannot connect");
          setError(new Error("Not authorized for cloud inference"));
          setIsReady(false);
          return;
        }
        token = dynamicToken;
      }

      console.log("[CloudProvider] Connecting to cloud:", wsUrl);
      const cloudAdapter = new CloudAdapter(wsUrl!, token);
      adapterRef.current = cloudAdapter;
      setAdapter(cloudAdapter);

      try {
        await cloudAdapter.connect();
        if (cancelled) {
          cloudAdapter.disconnect();
          return;
        }
        console.log("[CloudProvider] Connected to cloud");
        setIsReady(true);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        console.error("[CloudProvider] Connection failed:", err);
        setError(err as Error);
        setIsReady(false);
      }
    }

    connectWithToken();

    // Refresh token every 4 minutes (tokens last 5 min) and reconnect
    if (tokenProvider) {
      tokenRefreshRef.current = setInterval(
        async () => {
          const newToken = await tokenProvider();
          if (!newToken) {
            // Credits exhausted — disconnect
            console.log("[CloudProvider] Token refresh failed — disconnecting");
            adapterRef.current?.disconnect();
            setIsReady(false);
            setError(new Error("Credits exhausted"));
            window.dispatchEvent(new Event("billing:credits-exhausted"));
            return;
          }

          // Reconnect with new token
          const oldAdapter = adapterRef.current;
          oldAdapter?.disconnect();

          const cloudAdapter = new CloudAdapter(wsUrl!, newToken);
          adapterRef.current = cloudAdapter;
          setAdapter(cloudAdapter);

          try {
            await cloudAdapter.connect();
            setIsReady(true);
            setError(null);
          } catch (err) {
            console.error("[CloudProvider] Token refresh reconnect failed:", err);
            setError(err as Error);
            setIsReady(false);
          }
        },
        4 * 60 * 1000,
      );
    }

    return () => {
      cancelled = true;
      console.log("[CloudProvider] Disconnecting from cloud");
      adapterRef.current?.disconnect();
      adapterRef.current = null;
      if (tokenRefreshRef.current) {
        clearInterval(tokenRefreshRef.current);
        tokenRefreshRef.current = null;
      }
    };
  }, [wsUrl, apiKey, tokenProvider]);

  // Listen for credits-exhausted to disconnect immediately
  useEffect(() => {
    const handler = () => {
      console.log("[CloudProvider] Credits exhausted — disconnecting");
      adapterRef.current?.disconnect();
      setIsReady(false);
    };
    window.addEventListener("billing:credits-exhausted", handler);
    return () =>
      window.removeEventListener("billing:credits-exhausted", handler);
  }, []);

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
