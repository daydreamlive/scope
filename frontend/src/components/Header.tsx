import { useState, useEffect } from "react";
import { Settings } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";

interface HeaderProps {
  className?: string;
}

export function Header({ className = "" }: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<"general" | "plugins">(
    "general"
  );
  const [initialPluginPath, setInitialPluginPath] = useState("");

  useEffect(() => {
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
      />
    </header>
  );
}
