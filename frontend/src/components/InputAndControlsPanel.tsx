import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Button } from "./ui/button";
import { Play, Square, Loader2, Upload } from "lucide-react";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type { PromptItem } from "../lib/api";
import { PIPELINES } from "../data/pipelines";
import { PromptInput } from "./PromptInput";
import { TimelineCheckbox } from "./TimelineCheckbox";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import type { TimelinePrompt } from "./PromptTimeline";

interface InputAndControlsPanelProps {
  className?: string;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  mode: VideoSourceMode;
  onModeChange: (mode: VideoSourceMode) => void;
  isStreaming: boolean;
  isConnecting: boolean;
  isPipelineLoading: boolean;
  canStartStream: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  pipelineId: string;
  prompts: PromptItem[];
  onPromptsChange: (prompts: PromptItem[]) => void;
  onPromptsSubmit: (prompts: PromptItem[]) => void;
  interpolationMethod: "linear" | "slerp";
  onInterpolationMethodChange: (method: "linear" | "slerp") => void;
  showTimeline?: boolean;
  onShowTimelineChange?: (show: boolean) => void;
  isRecording?: boolean;
  onRecordingPromptSubmit?: (prompts: PromptItem[]) => void;
  selectedTimelinePrompt?: TimelinePrompt | null;
  onTimelinePromptUpdate?: (prompt: TimelinePrompt) => void;
}

export function InputAndControlsPanel({
  className = "",
  localStream,
  isInitializing,
  error,
  mode,
  onModeChange,
  isStreaming,
  isConnecting,
  isPipelineLoading,
  canStartStream,
  onStartStream,
  onStopStream,
  onVideoFileUpload,
  pipelineId,
  prompts,
  onPromptsChange,
  onPromptsSubmit,
  interpolationMethod,
  onInterpolationMethodChange,
  showTimeline = false,
  onShowTimelineChange,
  isRecording = false,
  onRecordingPromptSubmit,
  selectedTimelinePrompt = null,
  onTimelinePromptUpdate,
}: InputAndControlsPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Initialize recording prompt with current prompt when recording starts
  useEffect(() => {
    if (isRecording && prompts.length > 0) {
      // This is now handled by the PromptInput component
    }
  }, [isRecording, prompts]);

  // Get pipeline category, deafault to video-input
  const pipelineCategory = PIPELINES[pipelineId]?.category || "video-input";

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  const handleStreamClick = () => {
    if (isStreaming) {
      onStopStream();
    } else {
      onStartStream();
    }
  };

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
    // Reset the input value so the same file can be selected again
    event.target.value = "";
  };

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">
          Input & Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div>
          <h3 className="text-sm font-medium mb-2">Mode</h3>
          <Select
            value={pipelineCategory === "video-input" ? mode : "text"}
            onValueChange={value => {
              if (pipelineCategory === "video-input" && value) {
                onModeChange(value as VideoSourceMode);
              }
            }}
            disabled={isStreaming}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pipelineCategory === "video-input" ? (
                <>
                  <SelectItem value="video">Video</SelectItem>
                  <SelectItem value="camera">Camera</SelectItem>
                </>
              ) : (
                <SelectItem value="text">Text</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>

        {pipelineCategory === "video-input" && (
          <div>
            <h3 className="text-sm font-medium mb-2">Input</h3>
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
                <div className="text-center text-muted-foreground text-sm">
                  {mode === "camera" ? "Camera Preview" : "Video Preview"}
                </div>
              )}

              {/* Upload button - only show in video mode */}
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
          </div>
        )}

        <div>
          <h3 className="text-sm font-medium mb-2">Controls</h3>
          <div className="flex flex-wrap gap-2 min-w-0">
            <Button
              onClick={handleStreamClick}
              variant={isStreaming ? "destructive" : "default"}
              size="sm"
              disabled={
                isPipelineLoading ||
                isConnecting ||
                (!canStartStream && !isStreaming)
              }
              className="w-full gap-2"
            >
              {isPipelineLoading || isConnecting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : isStreaming ? (
                <Square className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {isPipelineLoading || isConnecting
                ? ""
                : isStreaming
                  ? "Stop"
                  : "Start"}
            </Button>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium mb-2">Prompts</h3>
          {selectedTimelinePrompt ? (
            <TimelinePromptEditor
              prompt={selectedTimelinePrompt}
              onPromptUpdate={onTimelinePromptUpdate}
              onPromptSubmit={onTimelinePromptUpdate}
              disabled={isRecording}
            />
          ) : (
            <PromptInput
              prompts={prompts}
              onPromptsChange={onPromptsChange}
              onPromptsSubmit={onPromptsSubmit}
              disabled={pipelineId === "passthrough" || pipelineId === "vod"}
              interpolationMethod={interpolationMethod}
              onInterpolationMethodChange={onInterpolationMethodChange}
              isRecording={isRecording}
              onRecordingPromptSubmit={onRecordingPromptSubmit}
            />
          )}
        </div>

        <div>
          <TimelineCheckbox
            checked={showTimeline}
            onChange={onShowTimelineChange || (() => {})}
            disabled={pipelineId === "passthrough" || pipelineId === "vod"}
          />
        </div>
      </CardContent>
    </Card>
  );
}
