import { useState, useEffect } from "react";
import { Settings } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";

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
    "general" | "api-keys" | "plugins"
  >("general");
  const [initialPluginPath, setInitialPluginPath] = useState("");

  // React to external requests to open a specific settings tab
  useEffect(() => {
    if (openSettingsTab) {
      setInitialTab(openSettingsTab as "general" | "api-keys" | "plugins");
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

      <SettingsDialog
        open={settingsOpen}
        onClose={handleClose}
        initialTab={initialTab}
        initialPluginPath={initialPluginPath}
        onCloudStatusChange={onCloudStatusChange}
        onCloudConnectingChange={onCloudConnectingChange}
        onPipelinesRefresh={onPipelinesRefresh}
        cloudDisabled={cloudDisabled}
      />
    </header>
  );
}
