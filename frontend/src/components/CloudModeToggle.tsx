/**
 * CloudModeToggle - Backend cloud mode integration
 *
 * This component manages cloud mode through the local scope backend,
 * which then connects to fal.ai. This is different from the FalProvider
 * which connects directly from the browser.
 *
 * When cloud mode is enabled:
 * - The local scope backend connects to fal.ai via WebSocket
 * - API calls (pipeline load, status) are proxied through fal
 * - WebRTC signaling is proxied through fal
 * - Video streams flow directly between browser and fal.ai
 */

import { useState, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Cloud, CloudOff, Loader2 } from "lucide-react";

interface CloudStatus {
  connected: boolean;
  app_id: string | null;
}

interface CloudModeToggleProps {
  className?: string;
  /** Callback when cloud mode status changes */
  onStatusChange?: (connected: boolean) => void;
}

// Local storage keys for persistence
const STORAGE_KEY_APP_ID = "scope-cloud-app-id";
const STORAGE_KEY_API_KEY = "scope-cloud-api-key";

export function CloudModeToggle({
  className,
  onStatusChange,
}: CloudModeToggleProps) {
  const [status, setStatus] = useState<CloudStatus>({
    connected: false,
    app_id: null,
  });
  const [appId, setAppId] = useState(
    () => localStorage.getItem(STORAGE_KEY_APP_ID) || ""
  );
  const [apiKey, setApiKey] = useState(
    () => localStorage.getItem(STORAGE_KEY_API_KEY) || ""
  );
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Persist credentials to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_APP_ID, appId);
  }, [appId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_API_KEY, apiKey);
  }, [apiKey]);

  // Poll status on mount and periodically
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/v1/fal/status");
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        onStatusChange?.(data.connected);
      }
    } catch (e) {
      console.error("[CloudModeToggle] Failed to fetch status:", e);
    }
  }, [onStatusChange]);

  useEffect(() => {
    fetchStatus();
    // Poll every 5 seconds to detect disconnections
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleConnect = async () => {
    if (!appId.trim() || !apiKey.trim()) {
      setError("Please enter both App ID and API Key");
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/fal/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_id: appId.trim(),
          api_key: apiKey.trim(),
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Connection failed");
      }

      const data = await response.json();
      setStatus(data);
      onStatusChange?.(data.connected);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Connection failed";
      setError(message);
      console.error("[CloudModeToggle] Connect failed:", e);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/fal/disconnect", {
        method: "POST",
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Disconnect failed");
      }

      const data = await response.json();
      setStatus(data);
      onStatusChange?.(data.connected);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Disconnect failed";
      setError(message);
      console.error("[CloudModeToggle] Disconnect failed:", e);
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          {status.connected ? (
            <Cloud className="h-4 w-4 text-green-500" />
          ) : (
            <CloudOff className="h-4 w-4 text-muted-foreground" />
          )}
          Cloud Mode
          {status.connected && (
            <Badge variant="secondary" className="ml-auto text-xs">
              Connected
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {!status.connected ? (
          <>
            <div className="space-y-2">
              <Input
                placeholder="App ID (e.g., username/scope-fal)"
                value={appId}
                onChange={(e) => setAppId(e.target.value)}
                disabled={isConnecting}
                className="text-sm"
              />
              <Input
                type="password"
                placeholder="fal API Key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={isConnecting}
                className="text-sm"
              />
            </div>
            <Button
              onClick={handleConnect}
              disabled={isConnecting || !appId.trim() || !apiKey.trim()}
              className="w-full"
              size="sm"
            >
              {isConnecting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  <Cloud className="mr-2 h-4 w-4" />
                  Connect to Cloud
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground">
              Connect to fal.ai for cloud GPU inference. Cold starts may take
              1-2 minutes.
            </p>
          </>
        ) : (
          <>
            <div className="text-sm text-muted-foreground truncate">
              {status.app_id}
            </div>
            <Button
              onClick={handleDisconnect}
              disabled={isConnecting}
              variant="outline"
              className="w-full"
              size="sm"
            >
              {isConnecting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Disconnecting...
                </>
              ) : (
                <>
                  <CloudOff className="mr-2 h-4 w-4" />
                  Disconnect
                </>
              )}
            </Button>
          </>
        )}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
