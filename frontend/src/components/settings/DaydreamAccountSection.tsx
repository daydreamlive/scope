/**
 * DaydreamAccountSection - Auth and Cloud Mode UI for Settings
 *
 * Displays:
 * - Not logged in: Sign in/Sign up buttons
 * - Logged in: User info, Manage/Log out buttons, Cloud Mode toggle
 * - Cloud connecting/connected states
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import { Cloud, Loader2, X, Copy, Check } from "lucide-react";
import {
  isAuthenticated,
  redirectToSignIn,
  clearDaydreamAuth,
  getDaydreamUserId,
} from "../../lib/auth";

interface CloudStatus {
  connected: boolean;
  app_id: string | null;
  connection_id: string | null;
  credentials_configured: boolean;
}

interface DaydreamAccountSectionProps {
  /** Callback when cloud mode status changes */
  onStatusChange?: (connected: boolean) => void;
  /** Callback when connecting state changes */
  onConnectingChange?: (connecting: boolean) => void;
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  disabled?: boolean;
}

export function DaydreamAccountSection({
  onStatusChange,
  onConnectingChange,
  onPipelinesRefresh,
  disabled = false,
}: DaydreamAccountSectionProps) {
  // Auth state
  const [isSignedIn, setIsSignedIn] = useState(false);

  // Cloud status state (same as CloudModeToggle)
  const [status, setStatus] = useState<CloudStatus>({
    connected: false,
    app_id: null,
    connection_id: null,
    credentials_configured: false,
  });
  const [isConnecting, setIsConnecting] = useState(false);
  const [isDisconnecting, setIsDisconnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Check auth on mount and listen for changes
  useEffect(() => {
    setIsSignedIn(isAuthenticated());

    const handleAuthChange = () => {
      setIsSignedIn(isAuthenticated());
    };

    window.addEventListener("daydream-auth-change", handleAuthChange);
    return () => {
      window.removeEventListener("daydream-auth-change", handleAuthChange);
    };
  }, []);

  // Poll cloud status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/v1/cloud/status");
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        onStatusChange?.(data.connected);
      }
    } catch (e) {
      console.error("[DaydreamAccountSection] Failed to fetch status:", e);
    }
  }, [onStatusChange]);

  useEffect(() => {
    if (isSignedIn) {
      fetchStatus();
      const interval = setInterval(fetchStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [isSignedIn, fetchStatus]);

  // Notify parent when connecting/disconnecting state changes
  useEffect(() => {
    onConnectingChange?.(isConnecting || isDisconnecting);
  }, [isConnecting, isDisconnecting, onConnectingChange]);

  const handleCopyConnectionId = async () => {
    if (status.connection_id) {
      try {
        await navigator.clipboard.writeText(status.connection_id);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (e) {
        console.error("[DaydreamAccountSection] Failed to copy:", e);
      }
    }
  };

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const userId = getDaydreamUserId();

      const response = await fetch("/api/v1/cloud/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Connection failed");
      }

      const data = await response.json();
      setStatus(data);
      onStatusChange?.(data.connected);

      if (data.connected && onPipelinesRefresh) {
        try {
          await onPipelinesRefresh();
        } catch (refreshError) {
          console.error(
            "[DaydreamAccountSection] Failed to refresh pipelines:",
            refreshError
          );
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        console.log("[DaydreamAccountSection] Connection canceled by user");
        return;
      }
      const message = e instanceof Error ? e.message : "Connection failed";
      setError(message);
      console.error("[DaydreamAccountSection] Connect failed:", e);
    } finally {
      abortControllerRef.current = null;
      setIsConnecting(false);
    }
  };

  const handleCancel = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    try {
      await fetch("/api/v1/cloud/disconnect", { method: "POST" });
      await fetchStatus();
    } catch (e) {
      console.error(
        "[DaydreamAccountSection] Failed to cleanup after cancel:",
        e
      );
    }

    setIsConnecting(false);
  };

  const handleDisconnect = async () => {
    setIsDisconnecting(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/cloud/disconnect", {
        method: "POST",
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Disconnect failed");
      }

      const data = await response.json();
      setStatus(data);
      onStatusChange?.(data.connected);

      if (!data.connected && onPipelinesRefresh) {
        try {
          await onPipelinesRefresh();
        } catch (refreshError) {
          console.error(
            "[DaydreamAccountSection] Failed to refresh pipelines:",
            refreshError
          );
        }
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Disconnect failed";
      setError(message);
      console.error("[DaydreamAccountSection] Disconnect failed:", e);
    } finally {
      setIsDisconnecting(false);
    }
  };

  const handleToggle = async (checked: boolean) => {
    if (checked) {
      await handleConnect();
    } else {
      await handleDisconnect();
    }
  };

  const handleSignIn = () => {
    redirectToSignIn();
  };

  const handleSignOut = () => {
    // Disconnect from cloud if connected before signing out
    if (status.connected) {
      handleDisconnect();
    }
    clearDaydreamAuth();
    setIsSignedIn(false);
  };

  return (
    <div className="rounded-lg bg-muted/50 p-4 space-y-4">
      <h3 className="text-sm font-medium text-foreground">Daydream Account</h3>

      {!isSignedIn ? (
        // Not logged in state
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">Not logged in</p>
          <Button onClick={handleSignIn} variant="default" size="sm">
            Log in
          </Button>
        </div>
      ) : (
        // Logged in state
        <div className="space-y-4">
          {/* User info and actions */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Signed in</span>
            <div className="flex gap-2">
              <Button onClick={handleSignOut} variant="outline" size="sm">
                Log out
              </Button>
            </div>
          </div>

          {/* Cloud Mode Toggle */}
          {status.credentials_configured ? (
            <div className="space-y-3 pt-2 border-t border-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cloud className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Cloud Mode</span>
                </div>
                {isConnecting ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      Connecting...
                    </span>
                    <Button
                      onClick={handleCancel}
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      title="Cancel connection"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ) : isDisconnecting ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      Disconnecting...
                    </span>
                  </div>
                ) : (
                  <Switch
                    checked={status.connected}
                    onCheckedChange={handleToggle}
                    disabled={disabled}
                    className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
                  />
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Use Daydream cloud models
              </p>

              {/* Connection ID when connected */}
              {status.connected && status.connection_id && (
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-xs text-muted-foreground">
                    Connection ID:{" "}
                    <code className="bg-background px-1 rounded">
                      {status.connection_id}
                    </code>
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0"
                    onClick={handleCopyConnectionId}
                    title="Copy connection ID"
                  >
                    {copied ? (
                      <Check className="h-3 w-3 text-green-500" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              )}

              {error && <p className="text-xs text-destructive">{error}</p>}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground pt-2 border-t border-border">
              Cloud mode requires the server to be started with cloud
              credentials configured.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
