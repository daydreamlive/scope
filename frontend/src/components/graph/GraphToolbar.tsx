import { useRef } from "react";
import { Play, Square } from "lucide-react";
import { NODE_TOKENS } from "./ui";
import { trackEvent } from "../../lib/analytics";

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
}: GraphToolbarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div data-tour="add-node" className={NODE_TOKENS.toolbar}>
      <button
        data-tour="play-button"
        onClick={isStreaming ? onStopStream : onStartStream}
        disabled={isConnecting || isLoading}
        className={`${NODE_TOKENS.toolbarButton} ${isConnecting || isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
        title={isStreaming ? "Stop stream" : "Start stream"}
      >
        {isConnecting || isLoading ? (
          <span className="inline-flex items-center gap-1">
            <svg
              className="animate-spin h-3 w-3"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
          </span>
        ) : isStreaming ? (
          <Square className="h-3.5 w-3.5" />
        ) : (
          <Play className="h-3.5 w-3.5" />
        )}
      </button>

      <div className="flex-1" />

      {status && <span className={NODE_TOKENS.toolbarStatus}>{status}</span>}

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
      <button
        onClick={() => fileInputRef.current?.click()}
        className={NODE_TOKENS.toolbarButton}
      >
        Import
      </button>
      <button onClick={() => { onExport(); trackEvent("workflow_exported", { surface: "graph_mode" }); }} className={NODE_TOKENS.toolbarButton}>
        Export
      </button>
      <button
        onClick={onClear}
        className={NODE_TOKENS.toolbarButton}
        title="Clear graph"
      >
        Clear
      </button>
    </div>
  );
}
