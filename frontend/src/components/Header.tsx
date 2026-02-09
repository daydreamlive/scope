import { useState, useEffect, useCallback, useRef } from "react";
import { Settings, Cloud, CloudOff } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";
import { toast } from "sonner";

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
  const [cloudConnected, setCloudConnected] = useState(false);
  const [cloudConnecting, setCloudConnecting] = useState(false);

  // Track the last close code we've shown a toast for to avoid duplicates
  const lastNotifiedCloseCodeRef = useRef<number | null>(null);

  // Fetch initial cloud status
  useEffect(() => {
    const fetchCloudStatus = async () => {
      try {
        const response = await fetch("/api/v1/cloud/status");
        if (response.ok) {
          const data = await response.json();
          const isConnected = data.connected;
          const isConnecting = data.connecting ?? false;

          // Detect unexpected disconnection: show toast when there's a close code we haven't
          // notified about yet. The presence of a close code means a WebSocket was connected
          // and then closed. Skip code 1000 (normal closure from manual disconnect).
          const closeCode = data.last_close_code;
          if (
            closeCode !== null &&
            closeCode !== lastNotifiedCloseCodeRef.current
          ) {
            const closeReason = data.last_close_reason;
            console.warn(
              `[Header] Cloud WebSocket closed unexpectedly (code=${closeCode}, reason=${closeReason})`
            );
            toast.error("Cloud connection lost", {
              description: `WebSocket closed (code: ${closeCode}${closeReason ? `, reason: ${closeReason}` : ""})`,
              duration: 10000,
            });
            lastNotifiedCloseCodeRef.current = closeCode;
          }

          // Update state
          setCloudConnected(isConnected);
          setCloudConnecting(isConnecting);

          // Reset the notified close code when connected (so we can show it again if it disconnects later)
          if (isConnected) {
            lastNotifiedCloseCodeRef.current = null;
          }
        }
      } catch (e) {
        console.error("[Header] Failed to fetch cloud status:", e);
      }
    };
    fetchCloudStatus();
    // Poll periodically
    const interval = setInterval(fetchCloudStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle cloud status changes from settings dialog
  const handleCloudStatusChange = useCallback(
    (connected: boolean) => {
      setCloudConnected(connected);
      onCloudStatusChange?.(connected);
    },
    [onCloudStatusChange]
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
              cloudConnected
                ? "text-green-500 opacity-80"
                : cloudConnecting
                  ? "text-amber-400 opacity-80"
                  : "text-muted-foreground opacity-60"
            }`}
            title={
              cloudConnected
                ? "Cloud connected"
                : cloudConnecting
                  ? "Connecting to cloud..."
                  : "Cloud disconnected"
            }
          >
            {cloudConnected ? (
              <Cloud className="h-5 w-5" />
            ) : cloudConnecting ? (
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
