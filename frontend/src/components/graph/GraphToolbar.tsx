import { useRef, type RefObject } from "react";
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
  RotateCcw,
} from "lucide-react";
import { NODE_TOKENS } from "./ui";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { trackEvent } from "../../lib/analytics";
import { getShortcutById } from "../../lib/shortcuts";
import { getEffectiveShortcuts } from "../../lib/shortcutOverrides";

interface GraphToolbarProps {
  isStreaming: boolean;
  isConnecting: boolean;
  isLoading: boolean;
  loadingStage?: string | null;
  status: string;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onImport: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onExport: () => void;
  onClear: () => void;
  onDefaultWorkflow?: () => void;
  onOpenSettings?: () => void;
  onOpenPlugins?: () => void;
  fileInputRef?: RefObject<HTMLInputElement | null>;
}

export function GraphToolbar({
  isStreaming,
  isConnecting,
  isLoading,
  loadingStage,
  status,
  onStartStream,
  onStopStream,
  onImport,
  onExport,
  onClear,
  onDefaultWorkflow,
  onOpenSettings,
  onOpenPlugins,
  fileInputRef: externalFileInputRef,
}: GraphToolbarProps) {
  const internalFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = externalFileInputRef ?? internalFileInputRef;
  const busy = isConnecting || isLoading;

  const shortcuts = getEffectiveShortcuts();
  const streamShortcut = getShortcutById("toggle-stream", shortcuts);
  const exportShortcut = getShortcutById("export", shortcuts);

  return (
    <TooltipProvider delayDuration={400}>
      <div data-tour="add-node" className={NODE_TOKENS.toolbar}>
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
            <DropdownMenuItem
              onSelect={() => {
                onExport();
                trackEvent("workflow_exported", { surface: "graph_mode" });
              }}
            >
              <Download className="h-4 w-4" />
              Export Workflow
              {exportShortcut && (
                <kbd className="ml-auto font-mono text-[10px] text-muted-foreground">
                  {exportShortcut.keys}
                </kbd>
              )}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={onOpenSettings}>
              <Settings className="h-4 w-4" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onOpenPlugins}>
              <Plug className="h-4 w-4" />
              Nodes
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={onDefaultWorkflow}>
              <RotateCcw className="h-4 w-4" />
              Default Workflow
            </DropdownMenuItem>
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
            trackEvent("workflow_imported", { surface: "graph_mode" });
          }}
          className="hidden"
        />

        {/* ── Spacer ── */}
        <div className="flex-1" />

        {/* ── Loading stage ── */}
        {isLoading && (
          <div className="flex items-center gap-2 mr-3 self-center">
            <span
              className="text-xs text-[#8c8c8d] animate-fade-in"
              key={loadingStage}
            >
              {loadingStage || "Loading pipeline…"}
            </span>
            <span className="text-[10px] text-[#b0b0b0]">
              Models may take up to a minute to load, only on the first run.
            </span>
          </div>
        )}

        {/* ── Status text ── */}
        {!isLoading && status && (
          <span className={NODE_TOKENS.toolbarStatus}>{status}</span>
        )}

        {/* ── Right: Hero Play / Stop button ── */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              data-tour="play-button"
              onClick={isStreaming ? onStopStream : onStartStream}
              disabled={busy}
              className={
                busy
                  ? NODE_TOKENS.toolbarHeroBusy
                  : isStreaming
                    ? NODE_TOKENS.toolbarHeroStop
                    : NODE_TOKENS.toolbarHeroRun
              }
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
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <span>{isStreaming ? "Stop stream" : "Start stream"}</span>
            {streamShortcut && (
              <kbd className="ml-1.5 inline-flex items-center rounded border border-border/50 bg-muted/50 px-1 py-0.5 font-mono text-[10px] text-muted-foreground">
                {streamShortcut.keys}
              </kbd>
            )}
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
