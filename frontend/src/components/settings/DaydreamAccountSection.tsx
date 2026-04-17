/**
 * DaydreamAccountSection - Auth and Cloud Mode UI for Settings
 *
 * Displays:
 * - Not logged in: Sign in/Sign up buttons
 * - Logged in: User info, Manage/Log out buttons, Cloud GPU selector
 * - Cloud connecting/connected states (inside CloudGpuSelector)
 */

import { useState, useEffect } from "react";
import { Button } from "../ui/button";
import { Cloud, Copy, Check } from "lucide-react";
import {
  isAuthenticated,
  redirectToSignIn,
  clearDaydreamAuth,
  getDaydreamUserDisplayName,
  refreshUserProfile,
} from "../../lib/auth";
import { useCloudStatus } from "../../hooks/useCloudStatus";
import { CloudGpuSelector } from "./CloudGpuSelector";

interface DaydreamAccountSectionProps {
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable interactive cloud controls (e.g., when streaming) */
  disabled?: boolean;
}

export function DaydreamAccountSection({
  onPipelinesRefresh,
  disabled = false,
}: DaydreamAccountSectionProps) {
  // Auth state — initialise eagerly so the toggle is never briefly disabled on mount
  const [isSignedIn, setIsSignedIn] = useState(() => isAuthenticated());
  const [displayName, setDisplayName] = useState<string | null>(() =>
    getDaydreamUserDisplayName()
  );

  // Use shared cloud status hook - avoids redundant polling with Header
  const { status, refresh: refreshStatus } = useCloudStatus();

  const [copied, setCopied] = useState(false);

  // Keep auth state in sync with storage changes and ensure display name is populated
  useEffect(() => {
    const authed = isAuthenticated();
    const cachedName = getDaydreamUserDisplayName();
    // Keep state consistent in case localStorage was updated since the lazy init
    setIsSignedIn(authed);
    setDisplayName(cachedName);

    // If signed in but no display name cached yet, kick off a profile refresh
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

  const handleSignIn = () => {
    redirectToSignIn();
  };

  const handleSignOut = async () => {
    // Disconnect from cloud if connected before signing out
    if (status.connected || status.connecting) {
      try {
        await fetch("/api/v1/cloud/disconnect", { method: "POST" });
        await refreshStatus();
      } catch (e) {
        console.error(
          "[DaydreamAccountSection] Disconnect on sign-out failed:",
          e
        );
      }
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
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Cloud className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Remote Inference</span>
          </div>
          <CloudGpuSelector
            disabled={disabled || !isSignedIn}
            onPipelinesRefresh={onPipelinesRefresh}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Use remote inference for running pipelines.
          {!isSignedIn &&
            !(status.connected || status.connecting) &&
            " Log in required."}
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
      </div>
    </div>
  );
}
