import { RefreshCw, Trash2, AlertTriangle } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import type { LoRAFileInfo } from "@/lib/api";

/**
 * Detect a Hugging Face file *preview page* URL. These look like
 * `https://huggingface.co/<repo>/blob/main/<file>` and are NOT raw
 * downloads — pasting them silently saves the HTML preview as the
 * weights file. The HF "Copy download link" button produces the
 * correct `/resolve/main/` form. We surface this as inline guidance
 * before submit (and the backend rejects them too).
 *
 * Also handles `hf.co`, the official Hugging Face short domain that
 * redirects to huggingface.co.
 */
const HUGGING_FACE_HOSTS = ["huggingface.co", "hf.co"];

function isHuggingFaceBlobUrl(raw: string): boolean {
  const trimmed = raw.trim();
  if (!trimmed) return false;
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return false;
  }
  const host = parsed.hostname.toLowerCase();
  const isHf = HUGGING_FACE_HOSTS.some(
    h => host === h || host.endsWith(`.${h}`)
  );
  return isHf && parsed.pathname.includes("/blob/");
}

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
  const isBlobUrl = isHuggingFaceBlobUrl(installUrl);

  const handleInstall = () => {
    const trimmed = installUrl.trim();
    if (!trimmed || isBlobUrl) return;
    onInstall(trimmed);
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
      <div className="rounded-lg bg-muted/50 p-4 space-y-2">
        <div className="flex items-center gap-2">
          <Input
            value={installUrl}
            onChange={e => onInstallUrlChange(e.target.value)}
            placeholder="LoRA URL (HuggingFace or CivitAI)"
            className="flex-1"
            onKeyDown={e => {
              if (e.key === "Enter") handleInstall();
            }}
            aria-invalid={isBlobUrl || undefined}
          />
          <Button
            onClick={handleInstall}
            variant="outline"
            size="sm"
            disabled={isInstalling || !installUrl.trim() || isBlobUrl}
          >
            {isInstalling ? "Installing..." : "Install"}
          </Button>
        </div>
        {isBlobUrl && (
          <div className="flex items-start gap-2 text-xs text-amber-400/90">
            <AlertTriangle className="h-3.5 w-3.5 mt-[1px] shrink-0" />
            <span>
              That looks like a Hugging Face <em>preview</em> page. On the
              LoRA&rsquo;s file page, click{" "}
              <strong>&ldquo;Copy download link&rdquo;</strong> and paste that
              URL instead (the correct one contains{" "}
              <code className="font-mono">/resolve/main/</code>, not{" "}
              <code className="font-mono">/blob/main/</code>).
            </span>
          </div>
        )}
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
                      {!lora.read_only && (
                        <Button
                          onClick={() => onDelete(lora.name)}
                          variant="ghost"
                          size="icon"
                          disabled={isDeleting}
                          className="text-destructive hover:text-destructive hover:bg-destructive/10"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
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
