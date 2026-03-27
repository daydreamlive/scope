import { useRef } from "react";
import {
  Play,
  Square,
  MoreVertical,
  Upload,
  Download,
  Trash2,
  Loader2,
  Settings,
  Plug,
} from "lucide-react";
import { NODE_TOKENS } from "./ui";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";

interface GraphToolbarProps {
  isStreaming: boolean;
  isConnecting: boolean;
  isLoading: boolean;
  status: string;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onImport: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onExport: () => void;
  onClear: () => void;
  onOpenSettings?: () => void;
  onOpenPlugins?: () => void;
}

export function GraphToolbar({
  isStreaming,
  isConnecting,
  isLoading,
  status,
  onStartStream,
  onStopStream,
  onImport,
  onExport,
  onClear,
  onOpenSettings,
  onOpenPlugins,
}: GraphToolbarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const busy = isConnecting || isLoading;

  return (
    <div className={NODE_TOKENS.toolbar}>
      {/* ── Left: Menu dropdown ── */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className={NODE_TOKENS.toolbarMenuButton}>
            <MoreVertical className="h-3.5 w-3.5" />
            Menu
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" sideOffset={6}>
          <DropdownMenuItem onSelect={() => fileInputRef.current?.click()}>
            <Upload className="h-4 w-4" />
            Import Workflow
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={onExport}>
            <Download className="h-4 w-4" />
            Export Workflow
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={onOpenSettings}>
            <Settings className="h-4 w-4" />
            Settings
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={onOpenPlugins}>
            <Plug className="h-4 w-4" />
            Plugins
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={onClear}
            className="text-red-400 focus:text-red-300"
          >
            <Trash2 className="h-4 w-4" />
            Clear Graph
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.scope-workflow.json"
        onChange={e => {
          if (isStreaming) onStopStream?.();
          onImport(e);
        }}
        className="hidden"
      />

      {/* ── Spacer ── */}
      <div className="flex-1" />

      {/* ── Status text ── */}
      {status && <span className={NODE_TOKENS.toolbarStatus}>{status}</span>}

      {/* ── Right: Hero Play / Stop button ── */}
      <button
        onClick={isStreaming ? onStopStream : onStartStream}
        disabled={busy}
        className={
          busy
            ? NODE_TOKENS.toolbarHeroBusy
            : isStreaming
              ? NODE_TOKENS.toolbarHeroStop
              : NODE_TOKENS.toolbarHeroRun
        }
        title={isStreaming ? "Stop stream" : "Start stream"}
      >
        {busy ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : isStreaming ? (
          <Square className="h-3.5 w-3.5" />
        ) : (
          <Play className="h-3.5 w-3.5" />
        )}
        {busy ? "Starting…" : isStreaming ? "Stop" : "Run"}
      </button>
    </div>
  );
}
