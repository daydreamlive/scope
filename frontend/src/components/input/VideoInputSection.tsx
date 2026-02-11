import { useEffect, useRef } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { ToggleGroup, ToggleGroupItem } from "../ui/toggle-group";
import { Input } from "../ui/input";
import { Upload } from "lucide-react";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import type { VideoSourceMode } from "../../hooks/useVideoSource";
import type { InputMode } from "../../types";

interface VideoInputSectionProps {
  inputMode: InputMode;
  isMultiMode: boolean;
  mode: VideoSourceMode;
  spoutAvailable: boolean;
  spoutReceiverName: string;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  isStreaming: boolean;
  isConnecting: boolean;
  onInputModeChange: (mode: InputMode) => void;
  onModeChange: (mode: VideoSourceMode) => void;
  onSpoutReceiverChange?: (name: string) => void;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
}

export function VideoInputSection({
  inputMode,
  isMultiMode,
  mode,
  spoutAvailable,
  spoutReceiverName,
  localStream,
  isInitializing,
  error,
  isStreaming,
  isConnecting,
  onInputModeChange,
  onModeChange,
  onSpoutReceiverChange,
  onVideoFileUpload,
}: VideoInputSectionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file && onVideoFileUpload) {
      try {
        await onVideoFileUpload(file);
      } catch (error) {
        console.error("Video upload failed:", error);
      }
    }
    event.target.value = "";
  };

  return (
    <>
      {/* Input Mode selector - only show for multi-mode pipelines */}
      {isMultiMode && (
        <div>
          <h3 className="text-sm font-medium mb-2">Input Mode</h3>
          <Select
            value={inputMode}
            onValueChange={value => {
              if (value) {
                onInputModeChange(value as InputMode);
              }
            }}
            disabled={isStreaming}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="text">Text</SelectItem>
              <SelectItem value="video">Video</SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Video Source toggle - only show when in video input mode */}
      {inputMode === "video" && (
        <div>
          <h3 className="text-sm font-medium mb-2">Video Source</h3>
          <ToggleGroup
            type="single"
            value={mode}
            onValueChange={value => {
              if (value) {
                onModeChange(value as VideoSourceMode);
              }
            }}
            className="justify-start"
          >
            <ToggleGroupItem value="video" aria-label="Video file">
              File
            </ToggleGroupItem>
            <ToggleGroupItem value="camera" aria-label="Camera">
              Camera
            </ToggleGroupItem>
            {spoutAvailable && (
              <ToggleGroupItem value="spout" aria-label="Spout Receiver">
                Spout Receiver
              </ToggleGroupItem>
            )}
          </ToggleGroup>
        </div>
      )}

      {/* Video preview - only show when in video input mode */}
      {inputMode === "video" && (
        <div>
          <h3 className="text-sm font-medium mb-2">Input</h3>
          {mode === "spout" ? (
            <div className="flex items-center gap-3">
              <LabelWithTooltip
                label="Sender Name"
                tooltip="The name of the sender to receive video from Spout-compatible apps like TouchDesigner, Resolume, OBS. Leave empty to receive from any sender."
                className="text-xs text-muted-foreground whitespace-nowrap"
              />
              <Input
                type="text"
                value={spoutReceiverName}
                onChange={e => onSpoutReceiverChange?.(e.target.value)}
                disabled={isStreaming}
                className="h-8 text-sm flex-1"
                placeholder="TDSyphonSpoutOut"
              />
            </div>
          ) : (
            <div className="rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative">
              {isInitializing ? (
                <div className="text-center text-muted-foreground text-sm">
                  {mode === "camera"
                    ? "Requesting camera access..."
                    : "Initializing video..."}
                </div>
              ) : error ? (
                <div className="text-center text-red-500 text-sm p-4">
                  <p>
                    {mode === "camera"
                      ? "Camera access failed:"
                      : "Video error:"}
                  </p>
                  <p className="text-xs mt-1">{error}</p>
                </div>
              ) : localStream ? (
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  playsInline
                />
              ) : (
                <div className="text-center text-muted-foreground text-sm p-4">
                  {mode === "camera" ? "Camera Preview" : "Video Preview"}
                </div>
              )}

              {mode === "video" && onVideoFileUpload && (
                <>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="video-upload"
                    disabled={isStreaming || isConnecting}
                  />
                  <label
                    htmlFor="video-upload"
                    className={`absolute bottom-2 right-2 p-2 rounded-full bg-black/50 transition-colors ${
                      isStreaming || isConnecting
                        ? "opacity-50 cursor-not-allowed"
                        : "hover:bg-black/70 cursor-pointer"
                    }`}
                  >
                    <Upload className="h-4 w-4 text-white" />
                  </label>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
