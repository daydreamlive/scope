import { useState, useEffect, useCallback, useRef } from "react";
import { Settings, Cloud, CloudOff } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";
import { toast } from "sonner";
import { useCloudStatus } from "../hooks/useCloudStatus";

interface HeaderProps {
  className?: string;
  // Cloud mode callbacks
  onCloudStatusChange?: (connected: boolean) => void;
  onCloudConnectingChange?: (connecting: boolean) => void;
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
  // External settings tab control
  openSettingsTab?: string | null;
  onSettingsTabOpened?: () => void;
}

export function Header({
  className = "",
  onCloudStatusChange,
  onCloudConnectingChange,
  onPipelinesRefresh,
  cloudDisabled,
  openSettingsTab,
  onSettingsTabOpened,
}: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<
    "general" | "account" | "api-keys" | "plugins"
  >("general");
  const [initialPluginPath, setInitialPluginPath] = useState("");

  // Use shared cloud status hook - single source of truth for polling
  const {
    isConnected,
    isConnecting,
    lastCloseCode,
    lastCloseReason,
    refresh: refreshCloudStatus,
  } = useCloudStatus();

  // Track the last close code we've shown a toast for to avoid duplicates
  const lastNotifiedCloseCodeRef = useRef<number | null>(null);

  // Track previous connection state to detect transitions for pipeline refresh
  const prevConnectedRef = useRef(false);

  // Detect unexpected disconnection and show toast
  useEffect(() => {
    if (
      lastCloseCode !== null &&
      lastCloseCode !== lastNotifiedCloseCodeRef.current
    ) {
      console.warn(
        `[Header] Cloud WebSocket closed unexpectedly (code=${lastCloseCode}, reason=${lastCloseReason})`
      );
      toast.error("Cloud connection lost", {
        description: `WebSocket closed (code: ${lastCloseCode}${lastCloseReason ? `, reason: ${lastCloseReason}` : ""})`,
        duration: 10000,
      });
      lastNotifiedCloseCodeRef.current = lastCloseCode;
    }

    // Reset the notified close code when connected (so we can show it again if it disconnects later)
    if (isConnected) {
      lastNotifiedCloseCodeRef.current = null;
    }
  }, [lastCloseCode, lastCloseReason, isConnected]);

  // Notify parent when cloud status changes
  useEffect(() => {
    onCloudStatusChange?.(isConnected);
  }, [isConnected, onCloudStatusChange]);

  useEffect(() => {
    onCloudConnectingChange?.(isConnecting);
  }, [isConnecting, onCloudConnectingChange]);

  // Refresh pipelines when cloud connection status changes
  // This ensures pipeline list updates even if settings dialog is closed
  useEffect(() => {
    if (prevConnectedRef.current !== isConnected) {
      // Connection status changed - refresh pipelines to get the right list
      onPipelinesRefresh?.().catch(e =>
        console.error(
          "[Header] Failed to refresh pipelines after cloud status change:",
          e
        )
      );
    }
    prevConnectedRef.current = isConnected;
  }, [isConnected, onPipelinesRefresh]);

  // Handle cloud status changes from settings dialog (manual refresh)
  const handleCloudStatusChange = useCallback(
    (connected: boolean) => {
      // Refresh to sync state after manual connect/disconnect
      refreshCloudStatus();
      onCloudStatusChange?.(connected);
    },
    [onCloudStatusChange, refreshCloudStatus]
  );

  const handleCloudIconClick = () => {
    setInitialTab("account");
    setSettingsOpen(true);
  };

  // React to external requests to open a specific settings tab
  useEffect(() => {
    if (openSettingsTab) {
      setInitialTab(
        openSettingsTab as "general" | "account" | "api-keys" | "plugins"
      );
      setSettingsOpen(true);
      onSettingsTabOpened?.();
    }
  }, [openSettingsTab, onSettingsTabOpened]);

  useEffect(() => {
    // Handle deep link actions for plugin installation
    if (window.scope?.onDeepLinkAction) {
      return window.scope.onDeepLinkAction(data => {
        if (data.action === "install-plugin" && data.package) {
          setInitialTab("plugins");
          setInitialPluginPath(data.package);
          setSettingsOpen(true);
        }
      });
    }
  }, []);

  const handleClose = () => {
    setSettingsOpen(false);
    setInitialTab("general");
    setInitialPluginPath("");
  };

  return (
    <header className={`w-full bg-background px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-medium text-foreground">Daydream Scope</h1>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleCloudIconClick}
            className={`hover:opacity-80 transition-opacity h-8 w-8 ${
              isConnected
                ? "text-green-500 opacity-80"
                : isConnecting
                  ? "text-amber-400 opacity-80"
                  : "text-muted-foreground opacity-60"
            }`}
            title={
              isConnected
                ? "Cloud connected"
                : isConnecting
                  ? "Connecting to cloud..."
                  : "Cloud disconnected"
            }
          >
            {isConnected ? (
              <Cloud className="h-5 w-5" />
            ) : isConnecting ? (
              <Cloud className="h-5 w-5 animate-pulse" />
            ) : (
              <CloudOff className="h-5 w-5" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSettingsOpen(true)}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-60 h-8 w-8"
            title="Settings"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>

      <SettingsDialog
        open={settingsOpen}
        onClose={handleClose}
        initialTab={initialTab}
        initialPluginPath={initialPluginPath}
        onCloudStatusChange={handleCloudStatusChange}
        onCloudConnectingChange={onCloudConnectingChange}
        onPipelinesRefresh={onPipelinesRefresh}
        cloudDisabled={cloudDisabled}
      />
    </header>
  );
}
