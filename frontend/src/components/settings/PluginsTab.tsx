import { FolderOpen, Trash2 } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import type { InstalledPlugin } from "@/types/settings";

interface PluginsTabProps {
  plugins: InstalledPlugin[];
  installPath: string;
  onInstallPathChange: (path: string) => void;
  onBrowse: () => void;
  onInstall: (pluginUrl: string) => void;
  onCheckUpdates: () => void;
  onDelete: (pluginId: string) => void;
}

// Check if running in Electron (file browsing supported)
const isElectron =
  typeof window !== "undefined" &&
  navigator.userAgent.toLowerCase().includes("electron");

export function PluginsTab({
  plugins,
  installPath,
  onInstallPathChange,
  onBrowse,
  onInstall,
  onCheckUpdates,
  onDelete,
}: PluginsTabProps) {
  const handleInstall = () => {
    if (installPath.trim()) {
      onInstall(installPath.trim());
    }
  };

  return (
    <div className="space-y-4">
      {/* Install & Updates Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Install Plugin */}
        <div className="flex items-center gap-2">
          <Input
            value={installPath}
            onChange={e => onInstallPathChange(e.target.value)}
            placeholder={
              isElectron
                ? "PyPI package name, Git URL or local path"
                : "PyPI package name or Git URL"
            }
            className="flex-1"
          />
          {isElectron && (
            <Button
              onClick={onBrowse}
              variant="outline"
              size="icon"
              className="h-8 w-8"
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
          )}
          <Button onClick={handleInstall} variant="outline" size="sm">
            Install
          </Button>
        </div>

        {/* Check for Updates */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground">Updates</span>
          <div className="flex-1 flex items-center justify-end">
            <Button onClick={onCheckUpdates} variant="outline" size="sm">
              Check for updates
            </Button>
          </div>
        </div>
      </div>

      {/* Installed Plugins Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-3">
        <h3 className="text-sm font-medium text-foreground">
          Installed Plugins
        </h3>
        {plugins.length === 0 ? (
          <p className="text-sm text-muted-foreground">No plugins installed</p>
        ) : (
          <div className="space-y-3">
            {plugins.map(plugin => (
              <div
                key={plugin.id}
                className="flex items-start justify-between p-3 rounded-md border border-border bg-card"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">
                      {plugin.name}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      v{plugin.version}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    by {plugin.author}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {plugin.description}
                  </p>
                </div>
                <Button
                  onClick={() => onDelete(plugin.id)}
                  variant="ghost"
                  size="icon"
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
