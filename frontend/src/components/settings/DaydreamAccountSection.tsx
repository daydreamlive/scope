/**
 * DaydreamAccountSection - Auth and Cloud Mode UI for Settings
 *
 * Displays:
 * - Not logged in: Sign in/Sign up buttons
 * - Logged in: User info, Manage/Log out buttons, Cloud Mode toggle
 * - Cloud connecting/connected states
 */

import { useState, useEffect, useRef } from "react";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import { Cloud, Copy, Check } from "lucide-react";
import {
  isAuthenticated,
  redirectToSignIn,
  clearDaydreamAuth,
  getDaydreamUserId,
  getDaydreamUserDisplayName,
  refreshUserProfile,
} from "../../lib/auth";
import { useCloudStatus } from "../../hooks/useCloudStatus";

interface DaydreamAccountSectionProps {
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  disabled?: boolean;
}

export function DaydreamAccountSection({
  onPipelinesRefresh,
  disabled = false,
}: DaydreamAccountSectionProps) {
  // Auth state
  const [isSignedIn, setIsSignedIn] = useState(false);
  const [displayName, setDisplayName] = useState<string | null>(null);

  // Use shared cloud status hook - avoids redundant polling with Header
  const { status, refresh: refreshStatus } = useCloudStatus();

  // Local action state
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

    // If signed in but no display name cached, refresh the profile
    if (authed && !cachedName) {
      refreshUserProfile();
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
      await refreshStatus();
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

      // Refresh status from shared hook
      await refreshStatus();

      if (onPipelinesRefresh) {
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

  const handleSignOut = async () => {
    // Disconnect from cloud if connected before signing out
    if (status.connected) {
      await handleDisconnect();
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

      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cloud className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Remote Inference</span>
          </div>
          <Switch
            checked={status.connected || status.connecting}
            onCheckedChange={handleToggle}
            disabled={disabled || !isSignedIn || isDisconnecting}
            className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Use remote inference for running pipelines.
          {!isSignedIn && " Log in required."}
        </p>

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
