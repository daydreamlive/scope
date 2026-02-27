import {
  AlertTriangle,
  ArrowUpSquare,
  FolderOpen,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import type { InstalledPlugin } from "@/types/settings";
import type { FailedPluginInfo } from "@/lib/api";

interface PluginsTabProps {
  plugins: InstalledPlugin[];
  failedPlugins?: FailedPluginInfo[];
  installPath: string;
  onInstallPathChange: (path: string) => void;
  onBrowse: () => void;
  onInstall: (pluginUrl: string) => void;
  onUpdate: (pluginName: string, packageSpec: string) => void;
  onDelete: (pluginName: string) => void;
  onReload: (pluginName: string) => void;
  isLoading?: boolean;
  isInstalling?: boolean;
  disabled?: boolean;
  hideInstall?: boolean;
}

// Check if running in Electron (file browsing supported)
const isElectron =
  typeof window !== "undefined" &&
  navigator.userAgent.toLowerCase().includes("electron");

// Transform plain Git host URLs to git+ format on paste
const transformGitUrl = (value: string): string => {
  const trimmed = value.trim();

  // Already has git+ prefix - no change needed
  if (trimmed.startsWith("git+")) {
    return trimmed;
  }

  // Check if it's a URL to a known git host
  const gitHosts = ["github.com", "gitlab.com", "bitbucket.org"];
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    const isGitHost = gitHosts.some(host => trimmed.includes(host));
    if (isGitHost) {
      return `git+${trimmed}`;
    }
  }

  return trimmed;
};

export function PluginsTab({
  plugins,
  failedPlugins = [],
  installPath,
  onInstallPathChange,
  onBrowse,
  onInstall,
  onUpdate,
  onDelete,
  onReload,
  isLoading = false,
  isInstalling = false,
  disabled = false,
  hideInstall = false,
}: PluginsTabProps) {
  const handleInstall = () => {
    if (installPath.trim()) {
      onInstall(transformGitUrl(installPath.trim()));
    }
  };

  return (
    <div className="space-y-4">
      {/* Install & Updates Section */}
      {!hideInstall && (
        <div className="rounded-lg bg-muted/50 p-4 space-y-4">
          {/* Install Plugin */}
          <div className="flex items-center gap-2">
            <Input
              value={installPath}
              onChange={e => onInstallPathChange(e.target.value)}
              placeholder="PyPI package name, Git URL or local path"
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
            <Button
              onClick={handleInstall}
              variant="outline"
              size="sm"
              disabled={disabled || isInstalling || !installPath.trim()}
            >
              {isInstalling ? "Installing..." : "Install"}
            </Button>
          </div>
        </div>
      )}

      {/* Failed plugins warning */}
      {failedPlugins.length > 0 && (
        <div className="rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-3 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0 mt-0.5" />
          <p className="text-sm text-yellow-500">
            {failedPlugins.length === 1
              ? "1 plugin failed to load."
              : `${failedPlugins.length} plugins failed to load.`}{" "}
            Contact the plugin developer for a fix, then reinstall.
          </p>
        </div>
      )}

      {/* Installed Plugins Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-3">
        <h3 className="text-sm font-medium text-foreground">
          Installed Plugins
        </h3>
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading plugins...</p>
        ) : plugins.length === 0 ? (
          <p className="text-sm text-muted-foreground">No plugins installed</p>
        ) : (
          <div className="space-y-3">
            {plugins.map(plugin => {
              const failure = failedPlugins.find(
                fp => fp.package_name === plugin.name
              );
              return (
                <div
                  key={plugin.name}
                  className={`flex items-start justify-between p-3 rounded-md border bg-card ${failure ? "border-yellow-500/50" : "border-border"}`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {plugin.name}
                      </span>
                      {plugin.version && (
                        <span className="text-xs text-muted-foreground">
                          v{plugin.version}
                        </span>
                      )}
                    </div>
                    {plugin.author && (
                      <p className="text-xs text-muted-foreground">
                        by {plugin.author}
                      </p>
                    )}
                    {plugin.description && (
                      <p className="text-sm text-muted-foreground">
                        {plugin.description}
                      </p>
                    )}
                    {failure && (
                      <div className="flex items-start gap-1.5 text-yellow-500 mt-1">
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                        <span
                          className="text-xs line-clamp-2"
                          title={`${failure.error_type}: ${failure.error_message}`}
                        >
                          Failed to load &mdash; {failure.error_type}:{" "}
                          {failure.error_message}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {plugin.update_available && plugin.package_spec && (
                      <Button
                        onClick={() =>
                          onUpdate(plugin.name, plugin.package_spec!)
                        }
                        variant="ghost"
                        size="icon"
                        disabled={disabled || isInstalling}
                        title={
                          plugin.latest_version
                            ? `Update to ${plugin.latest_version}`
                            : "Update available"
                        }
                      >
                        <ArrowUpSquare className="h-4 w-4" />
                      </Button>
                    )}
                    {plugin.editable && (
                      <Button
                        onClick={() => onReload(plugin.name)}
                        variant="ghost"
                        size="icon"
                        disabled={disabled || isInstalling}
                        title="Reload plugin (restarts server)"
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      onClick={() => onDelete(plugin.name)}
                      variant="ghost"
                      size="icon"
                      disabled={disabled || isInstalling}
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
