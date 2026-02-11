import { Button } from "../ui/button";
import {
  Play,
  Pause,
  Circle,
  Download,
  Upload,
  RotateCcw,
  ChevronUp,
  ChevronDown,
  Trash2,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { ExportDialog } from "../ExportDialog";

interface TimelineHeaderProps {
  isPlaying: boolean;
  isCollapsed: boolean;
  disabled: boolean;
  isLoading: boolean;
  isDownloading: boolean;
  isStreaming: boolean;
  isRecording: boolean;
  videoScaleMode: "fit" | "native";
  showExportDialog: boolean;
  onPlayPause?: () => void;
  onReset?: () => void;
  onRecordingToggle?: () => void;
  onClear?: () => void;
  onVideoScaleModeToggle?: () => void;
  onExport: () => void;
  onCloseExportDialog: () => void;
  onSaveGeneration?: () => void;
  onSaveTimeline: () => void;
  onImport: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onCollapseToggle?: (collapsed: boolean) => void;
}

export function TimelineHeader({
  isPlaying,
  isCollapsed,
  disabled,
  isLoading,
  isDownloading,
  isStreaming,
  isRecording,
  videoScaleMode,
  showExportDialog,
  onPlayPause,
  onReset,
  onRecordingToggle,
  onClear,
  onVideoScaleModeToggle,
  onExport,
  onCloseExportDialog,
  onSaveGeneration,
  onSaveTimeline,
  onImport,
  onCollapseToggle,
}: TimelineHeaderProps) {
  return (
    <div
      className={`flex items-center justify-between ${isCollapsed ? "mb-0" : "mb-4"}`}
    >
      <div className="flex items-center gap-2">
        <Button
          onClick={onPlayPause}
          disabled={disabled || isLoading || isDownloading}
          size="sm"
          variant="outline"
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>
        <Button
          onClick={onReset}
          disabled={disabled || isLoading || isDownloading}
          size="sm"
          variant="outline"
          title="Reset timeline"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
        <Button
          onClick={onRecordingToggle}
          disabled={disabled || isLoading || isDownloading || isStreaming}
          size="sm"
          variant="outline"
          title={isRecording ? "Stop recording" : "Start recording"}
          className={
            isRecording
              ? "border-red-500 hover:border-red-400 animate-record-pulse"
              : ""
          }
        >
          <Circle
            className={`h-3.5 w-3.5 ${isRecording ? "fill-red-500 text-red-500" : "fill-muted-foreground text-muted-foreground"}`}
          />
        </Button>
        <Button
          onClick={onClear}
          disabled={
            disabled || isPlaying || isStreaming || isLoading || isDownloading
          }
          size="sm"
          variant="outline"
          title="Clear timeline"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex items-center gap-2">
        {onVideoScaleModeToggle && (
          <Button
            onClick={onVideoScaleModeToggle}
            size="sm"
            variant="outline"
            disabled={!isStreaming}
            title={
              videoScaleMode === "fit"
                ? "Switch to native resolution"
                : "Switch to fit to window"
            }
          >
            {videoScaleMode === "fit" ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
        )}
        <Button
          onClick={onExport}
          disabled={disabled || isLoading || isDownloading}
          size="sm"
          variant="outline"
        >
          <Upload className="h-4 w-4 mr-1" />
          Export
        </Button>
        <ExportDialog
          open={showExportDialog}
          onClose={onCloseExportDialog}
          onSaveGeneration={() => onSaveGeneration?.()}
          onSaveTimeline={onSaveTimeline}
          isRecording={isRecording}
        />
        <div className="relative">
          <input
            type="file"
            accept=".json"
            onChange={onImport}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={disabled || isStreaming || isLoading || isDownloading}
          />
          <Button
            size="sm"
            variant="outline"
            disabled={disabled || isStreaming || isLoading || isDownloading}
          >
            <Download className="h-4 w-4 mr-1" />
            Import
          </Button>
        </div>
        <Button
          onClick={() => onCollapseToggle?.(!isCollapsed)}
          size="sm"
          variant="outline"
          disabled={isLoading || isDownloading}
          title={isCollapsed ? "Expand timeline" : "Collapse timeline"}
        >
          {isCollapsed ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronUp className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
