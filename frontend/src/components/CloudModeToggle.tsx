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
 * - Video streams flow through the backend to fal.ai
 *
 * Credentials (app_id and api_key) are set via backend command line args.
 */

import { useState, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Cloud, CloudOff, Loader2 } from "lucide-react";

interface CloudStatus {
  connected: boolean;
  app_id: string | null;
  credentials_configured: boolean;
}

interface CloudModeToggleProps {
  className?: string;
  /** Callback when cloud mode status changes */
  onStatusChange?: (connected: boolean) => void;
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  disabled?: boolean;
}

export function CloudModeToggle({
  className,
  onStatusChange,
  onPipelinesRefresh,
  disabled = false,
}: CloudModeToggleProps) {
  const [status, setStatus] = useState<CloudStatus>({
    connected: false,
    app_id: null,
    credentials_configured: false,
  });
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setIsConnecting(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/fal/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Connection failed");
      }

      const data = await response.json();
      setStatus(data);
      onStatusChange?.(data.connected);

      // Refresh pipeline list to get cloud-available pipelines
      if (data.connected && onPipelinesRefresh) {
        try {
          await onPipelinesRefresh();
        } catch (refreshError) {
          console.error("[CloudModeToggle] Failed to refresh pipelines:", refreshError);
        }
      }
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

      // Refresh pipeline list to get local pipelines
      if (!data.connected && onPipelinesRefresh) {
        try {
          await onPipelinesRefresh();
        } catch (refreshError) {
          console.error("[CloudModeToggle] Failed to refresh pipelines:", refreshError);
        }
      }
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
            {status.credentials_configured ? (
              <>
                <Button
                  onClick={handleConnect}
                  disabled={isConnecting || disabled}
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
              <p className="text-xs text-muted-foreground">
                Cloud mode requires --fal-app-id and --fal-api-key command line
                arguments to be set when starting the server.
              </p>
            )}
          </>
        ) : (
          <>
            <div className="text-sm text-muted-foreground truncate">
              {status.app_id}
            </div>
            <Button
              onClick={handleDisconnect}
              disabled={isConnecting || disabled}
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
