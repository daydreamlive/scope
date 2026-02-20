import { RefreshCw, Trash2 } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import type { LoRAFileInfo } from "@/lib/api";

interface LoRAsTabProps {
  loraFiles: LoRAFileInfo[];
  installUrl: string;
  onInstallUrlChange: (url: string) => void;
  onInstall: (url: string) => void;
  onDelete: (name: string) => void;
  onRefresh: () => void;
  isLoading?: boolean;
  isInstalling?: boolean;
  deletingLoRAs?: Set<string>;
}

export function LoRAsTab({
  loraFiles,
  installUrl,
  onInstallUrlChange,
  onInstall,
  onDelete,
  onRefresh,
  isLoading = false,
  isInstalling = false,
  deletingLoRAs = new Set(),
}: LoRAsTabProps) {
  const handleInstall = () => {
    if (installUrl.trim()) {
      onInstall(installUrl.trim());
    }
  };

  // Group LoRA files by folder
  const groupedLoRAs = loraFiles.reduce(
    (acc, lora) => {
      const folder = lora.folder || "Root";
      if (!acc[folder]) {
        acc[folder] = [];
      }
      acc[folder].push(lora);
      return acc;
    },
    {} as Record<string, LoRAFileInfo[]>
  );

  const sortedFolders = Object.keys(groupedLoRAs).sort((a, b) => {
    if (a === "Root") return -1;
    if (b === "Root") return 1;
    return a.localeCompare(b);
  });

  return (
    <div className="space-y-4">
      {/* Install Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        <div className="flex items-center gap-2">
          <Input
            value={installUrl}
            onChange={e => onInstallUrlChange(e.target.value)}
            placeholder="LoRA URL (HuggingFace or CivitAI)"
            className="flex-1"
            onKeyDown={e => {
              if (e.key === "Enter") handleInstall();
            }}
          />
          <Button
            onClick={handleInstall}
            variant="outline"
            size="sm"
            disabled={isInstalling || !installUrl.trim()}
          >
            {isInstalling ? "Installing..." : "Install"}
          </Button>
        </div>
      </div>

      {/* Installed LoRAs Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-foreground">
            Installed LoRAs
          </h3>
          <Button
            onClick={onRefresh}
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            disabled={isLoading}
            title="Refresh LoRA list"
          >
            <RefreshCw
              className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>

        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading LoRAs...</p>
        ) : loraFiles.length === 0 ? (
          <div className="text-sm text-muted-foreground space-y-2">
            <p>No LoRA files found.</p>
            <p>
              Install LoRAs using the URL input above, or follow the{" "}
              <a
                href="https://docs.daydream.live/scope/guides/loras"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                documentation
              </a>{" "}
              for manual installation.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {sortedFolders.map(folder => (
              <div key={folder} className="space-y-2">
                {sortedFolders.length > 1 && (
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    {folder}
                  </h4>
                )}
                {groupedLoRAs[folder].map(lora => {
                  const isDeleting = deletingLoRAs.has(lora.name);
                  return (
                    <div
                      key={lora.path}
                      className="flex items-center justify-between p-3 rounded-md border bg-card"
                    >
                      <div className="space-y-0.5 min-w-0 flex-1">
                        <span className="text-sm font-medium text-foreground block truncate">
                          {lora.name}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {lora.size_mb.toFixed(1)} MB
                        </span>
                      </div>
                      <Button
                        onClick={() => onDelete(lora.name)}
                        variant="ghost"
                        size="icon"
                        disabled={isDeleting}
                        className="text-destructive hover:text-destructive hover:bg-destructive/10"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
