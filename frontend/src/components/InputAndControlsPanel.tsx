import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Upload } from "lucide-react";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type { PromptItem } from "../lib/api";
import { PIPELINES } from "../data/pipelines";
import { PromptInput } from "./PromptInput";
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
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  selectedTimelinePrompt?: TimelinePrompt | null;
  onTimelinePromptUpdate?: (prompt: TimelinePrompt) => void;
  isVideoPaused?: boolean;
  isTimelinePlaying?: boolean;
  currentTime?: number;
  timelinePrompts?: TimelinePrompt[];
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
  isPipelineLoading: _isPipelineLoading,
  canStartStream: _canStartStream,
  onStartStream: _onStartStream,
  onStopStream: _onStopStream,
  onVideoFileUpload,
  pipelineId,
  prompts,
  onPromptsChange,
  onPromptsSubmit,
  interpolationMethod,
  onInterpolationMethodChange,
  isLive = false,
  onLivePromptSubmit,
  selectedTimelinePrompt = null,
  onTimelinePromptUpdate,
  isVideoPaused = false,
  isTimelinePlaying: _isTimelinePlaying = false,
  currentTime: _currentTime = 0,
  timelinePrompts: _timelinePrompts = [],
}: InputAndControlsPanelProps) {
  // Helper function to determine if playhead is at the end of timeline
  const isAtEndOfTimeline = () => {
    if (_timelinePrompts.length === 0) return true;

    const sortedPrompts = [..._timelinePrompts].sort(
      (a, b) => a.endTime - b.endTime
    );
    const lastPrompt = sortedPrompts[sortedPrompts.length - 1];

    // Check if current time is at or past the end of the last prompt
    return _currentTime >= lastPrompt.endTime;
  };
  const videoRef = useRef<HTMLVideoElement>(null);

  // Initialize live prompt with current prompt when live mode starts
  useEffect(() => {
    if (isLive && prompts.length > 0) {
      // This is now handled by the PromptInput component
    }
  }, [isLive, prompts]);

  // Get pipeline category, deafault to video-input
  const pipelineCategory = PIPELINES[pipelineId]?.category || "video-input";

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
          <h3 className="text-sm font-medium mb-2">Prompts</h3>
          {(() => {
            // Simplified logic: Only two states - Append and Edit
            const isEditMode = selectedTimelinePrompt && isVideoPaused;

            return (
              <div>
                {/* Panel state indicator - only show in Edit Mode */}
                {isEditMode && (
                  <div className="text-xs text-gray-400 mb-2 px-2 py-1 bg-gray-50 rounded">
                    Edit Mode
                  </div>
                )}

                {selectedTimelinePrompt ? (
                  <TimelinePromptEditor
                    prompt={selectedTimelinePrompt}
                    onPromptUpdate={onTimelinePromptUpdate}
                    onPromptSubmit={onTimelinePromptUpdate}
                    disabled={false}
                  />
                ) : (
                  <PromptInput
                    prompts={prompts}
                    onPromptsChange={onPromptsChange}
                    onPromptsSubmit={onPromptsSubmit}
                    disabled={
                      pipelineId === "passthrough" ||
                      pipelineId === "vod" ||
                      (_isTimelinePlaying &&
                        !isVideoPaused &&
                        !isAtEndOfTimeline())
                    }
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    isLive={isLive}
                    onLivePromptSubmit={onLivePromptSubmit}
                  />
                )}
              </div>
            );
          })()}
        </div>
      </CardContent>
    </Card>
  );
}
