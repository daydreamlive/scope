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
import { Cloud, Copy, Check } from "lucide-react";
import {
  isAuthenticated,
  redirectToSignIn,
  clearDaydreamAuth,
  getDaydreamAPIKey,
  getDaydreamUserId,
  getDaydreamUserDisplayName,
  fetchAndStoreUserProfile,
} from "../../lib/auth";

interface CloudStatus {
  connected: boolean;
  connecting: boolean;
  error: string | null;
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
  const [displayName, setDisplayName] = useState<string | null>(null);

  // Cloud status state (same as CloudModeToggle)
  const [status, setStatus] = useState<CloudStatus>({
    connected: false,
    connecting: false,
    error: null,
    app_id: null,
    connection_id: null,
    credentials_configured: false,
  });
  const [isDisconnecting, setIsDisconnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const prevConnectedRef = useRef(false);

  // Check auth on mount and listen for changes
  useEffect(() => {
    const authed = isAuthenticated();
    const cachedName = getDaydreamUserDisplayName();
    setIsSignedIn(authed);
    setDisplayName(cachedName);

    // If signed in but no display name cached, fetch the profile
    if (authed && !cachedName) {
      const apiKey = getDaydreamAPIKey();
      if (apiKey) {
        fetchAndStoreUserProfile(apiKey);
      }
    }

    const handleAuthChange = () => {
      setIsSignedIn(isAuthenticated());
      setDisplayName(getDaydreamUserDisplayName());
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
    onConnectingChange?.(status.connecting || isDisconnecting);
  }, [status.connecting, isDisconnecting, onConnectingChange]);

  // Detect connection completion (connecting → connected) to trigger pipeline refresh
  useEffect(() => {
    if (!prevConnectedRef.current && status.connected) {
      // Just transitioned to connected
      onPipelinesRefresh?.().catch(e =>
        console.error(
          "[DaydreamAccountSection] Failed to refresh pipelines:",
          e
        )
      );
    }
    prevConnectedRef.current = status.connected;
  }, [status.connected, onPipelinesRefresh]);

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
    setError(null);

    try {
      const userId = getDaydreamUserId();

      const response = await fetch("/api/v1/cloud/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Connection failed");
      }

      // Backend returns immediately with connecting=true
      await fetchStatus();
    } catch (e) {
      const message = e instanceof Error ? e.message : "Connection failed";
      setError(message);
      console.error("[DaydreamAccountSection] Connect failed:", e);
    }
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

      {/* Auth row */}
      <div className="flex items-center justify-between">
        {isSignedIn ? (
          <>
            <span className="text-sm text-muted-foreground">
              {displayName || "Signed in"}
            </span>
            <Button onClick={handleSignOut} variant="outline" size="sm">
              Log out
            </Button>
          </>
        ) : (
          <>
            <span className="text-sm text-muted-foreground">Not logged in</span>
            <Button onClick={handleSignIn} variant="default" size="sm">
              Log in
            </Button>
          </>
        )}
      </div>

      {/* Cloud Mode section - always visible */}
      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cloud className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Cloud Mode</span>
          </div>
          <Switch
            checked={status.connected || status.connecting}
            onCheckedChange={handleToggle}
            disabled={disabled || !isSignedIn || isDisconnecting}
            className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Use Daydream Cloud for running pipelines.
          {!isSignedIn && " Log in required."}
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

        {(error || status.error) && (
          <p className="text-xs text-destructive">{error || status.error}</p>
        )}
      </div>
    </div>
  );
}
