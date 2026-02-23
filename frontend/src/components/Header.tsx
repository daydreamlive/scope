import { useState, useEffect, useRef } from "react";
import { Settings, Cloud, CloudOff, Plug } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";
import { PluginsDialog } from "./PluginsDialog";
import { toast } from "sonner";
import { useCloudStatus } from "../hooks/useCloudStatus";

interface HeaderProps {
  className?: string;
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
  // External settings tab control
  openSettingsTab?: string | null;
  onSettingsTabOpened?: () => void;
}

export function Header({
  className = "",
  onPipelinesRefresh,
  cloudDisabled,
  openSettingsTab,
  onSettingsTabOpened,
}: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [pluginsOpen, setPluginsOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<
    "general" | "account" | "api-keys" | "loras"
  >("general");
  const [initialPluginPath, setInitialPluginPath] = useState("");

  // Use shared cloud status hook - single source of truth
  const { isConnected, isConnecting, lastCloseCode, lastCloseReason } =
    useCloudStatus();

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

  const handleCloudIconClick = () => {
    setInitialTab("account");
    setSettingsOpen(true);
  };

  // React to external requests to open a specific settings/plugins tab
  useEffect(() => {
    if (openSettingsTab) {
      if (openSettingsTab === "plugins") {
        setPluginsOpen(true);
      } else {
        setInitialTab(
          openSettingsTab as "general" | "account" | "api-keys" | "loras"
        );
        setSettingsOpen(true);
      }
      onSettingsTabOpened?.();
    }
  }, [openSettingsTab, onSettingsTabOpened]);

  useEffect(() => {
    // Handle deep link actions for plugin installation
    if (window.scope?.onDeepLinkAction) {
      return window.scope.onDeepLinkAction(data => {
        if (data.action === "install-plugin" && data.package) {
          setInitialPluginPath(data.package);
          setPluginsOpen(true);
        }
      });
    }
  }, []);

  const handleSettingsClose = () => {
    setSettingsOpen(false);
    setInitialTab("general");
  };

  const handlePluginsClose = () => {
    setPluginsOpen(false);
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
            onClick={() => setPluginsOpen(true)}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-60 h-8 w-8"
            title="Plugins"
          >
            <Plug className="h-5 w-5" />
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

      <PluginsDialog
        open={pluginsOpen}
        onClose={handlePluginsClose}
        initialPluginPath={initialPluginPath}
      />

      <SettingsDialog
        open={settingsOpen}
        onClose={handleSettingsClose}
        initialTab={initialTab}
        onPipelinesRefresh={onPipelinesRefresh}
        cloudDisabled={cloudDisabled}
      />
    </header>
  );
}
