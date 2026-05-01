import { useState, useEffect, useCallback } from "react";
import { BookOpenText, Loader2, Copy, Check } from "lucide-react";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import { openExternalUrl } from "@/lib/openExternal";
import {
  fetchOscPaths,
  updateOscSettings,
  type OscInventoryEntry,
  type OscPathsResponse,
  type OscStatusResponse,
} from "@/lib/api";

interface OscTabProps {
  isActive: boolean;
}

export function OscTab({ isActive }: OscTabProps) {
  const [status, setStatus] = useState<OscStatusResponse | null>(null);
  const [paths, setPaths] = useState<OscPathsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTogglingLog, setIsTogglingLog] = useState(false);
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/v1/osc/status");
      if (res.ok) {
        setStatus(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch OSC status:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshPaths = useCallback(async () => {
    try {
      setPaths(await fetchOscPaths());
    } catch (err) {
      console.error("Failed to fetch OSC paths:", err);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchStatus();
      refreshPaths();
    }
  }, [isActive, fetchStatus, refreshPaths]);

  const handleCopyAddress = async (address: string) => {
    try {
      await navigator.clipboard.writeText(address);
      setCopiedAddress(address);
      window.setTimeout(() => setCopiedAddress(null), 1200);
    } catch (err) {
      console.warn("Clipboard write failed:", err);
    }
  };

  const handleOpenDocs = () => {
    openExternalUrl(`${window.location.origin}/api/v1/osc/docs`);
  };

  const handleToggleLogging = async (checked: boolean) => {
    setIsTogglingLog(true);
    try {
      const updated = await updateOscSettings({ log_all_messages: checked });
      setStatus(prev => (prev ? { ...prev, ...updated } : prev));
    } catch (err) {
      console.error("Failed to update OSC logging setting:", err);
    } finally {
      setIsTogglingLog(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Status */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Status
          </span>
          <div className="flex-1 flex items-center justify-end">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            ) : status?.listening ? (
              <span className="text-sm text-green-500">
                Listening on UDP port {status.port}
              </span>
            ) : (
              <span className="text-sm text-muted-foreground">
                Not listening
              </span>
            )}
          </div>
        </div>

        {/* Port */}
        {status?.port && (
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-foreground w-32">
              UDP Port
            </span>
            <div className="flex-1 flex items-center justify-end">
              <span className="text-sm text-muted-foreground font-mono">
                {status.port}
              </span>
            </div>
          </div>
        )}

        {/* Log all messages toggle */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Log Messages
          </span>
          <div className="flex-1 flex items-center justify-end gap-2">
            <span className="text-xs text-muted-foreground">
              {status?.log_all_messages ? "All" : "Errors only"}
            </span>
            <Switch
              aria-label="Log all OSC messages"
              checked={status?.log_all_messages ?? false}
              onCheckedChange={handleToggleLogging}
              disabled={isTogglingLog || isLoading || !status?.listening}
              className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
            />
          </div>
        </div>

        {/* Docs */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Reference
          </span>
          <div className="flex-1 flex items-center justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={handleOpenDocs}
              className="gap-1.5"
            >
              <BookOpenText className="h-4 w-4" />
              Open OSC Docs
            </Button>
          </div>
        </div>

        {/* Quick-start hint */}
        <div className="text-xs text-muted-foreground leading-relaxed">
          Send OSC messages to{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            /scope/&lt;param&gt;
          </code>{" "}
          on UDP port{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            {status?.port ?? "..."}
          </code>{" "}
          to control pipeline parameters in real time. Right-click any node →
          <strong> Configure OSC…</strong> to expose its params, or click{" "}
          <strong>Open OSC Docs</strong> for the full reference.
        </div>
      </div>

      <LiveInventoryPanel
        paths={paths}
        copiedAddress={copiedAddress}
        onCopy={handleCopyAddress}
        onRefresh={refreshPaths}
      />
    </div>
  );
}

interface LiveInventoryPanelProps {
  paths: OscPathsResponse | null;
  copiedAddress: string | null;
  onCopy: (address: string) => void;
  onRefresh: () => void;
}

function LiveInventoryPanel({
  paths,
  copiedAddress,
  onCopy,
  onRefresh,
}: LiveInventoryPanelProps) {
  const groups = paths?.active ?? {};
  const total = Object.values(groups).reduce(
    (sum, list) => sum + list.length,
    0
  );

  return (
    <div className="rounded-lg bg-muted/50 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-foreground">
          Currently exposed paths
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          className="h-7 text-xs"
        >
          Refresh
        </Button>
      </div>

      {total === 0 ? (
        <p className="text-xs text-muted-foreground">
          No paths exposed yet. Open the graph editor, right-click a node, and
          choose <em>Configure OSC…</em> to opt params in.
        </p>
      ) : (
        <div className="space-y-3">
          {Object.entries(groups).map(([groupName, entries]) => (
            <InventoryGroup
              key={groupName}
              name={groupName}
              entries={entries}
              copiedAddress={copiedAddress}
              onCopy={onCopy}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface InventoryGroupProps {
  name: string;
  entries: OscInventoryEntry[];
  copiedAddress: string | null;
  onCopy: (address: string) => void;
}

function InventoryGroup({
  name,
  entries,
  copiedAddress,
  onCopy,
}: InventoryGroupProps) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-1">
        {name}
      </div>
      <div className="space-y-1">
        {entries.map(entry => (
          <div
            key={`${name}:${entry.osc_address}`}
            className="flex items-center gap-2 text-xs"
          >
            <button
              type="button"
              onClick={() => onCopy(entry.osc_address)}
              className="font-mono text-foreground hover:underline truncate"
              title="Click to copy"
            >
              {entry.osc_address}
            </button>
            <span className="text-muted-foreground">{entry.type}</span>
            {entry.default !== undefined && entry.default !== null && (
              <span className="text-muted-foreground/70">
                default {String(entry.default)}
              </span>
            )}
            {copiedAddress === entry.osc_address ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3 text-muted-foreground/50" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
