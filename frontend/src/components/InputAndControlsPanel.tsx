import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Upload, ArrowUp } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { InputMode } from "../types";
import { pipelineIsMultiMode } from "../data/pipelines";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import type { TimelinePrompt } from "./PromptTimeline";
import { ImageManager } from "./ImageManager";
import { SliderWithInput } from "./ui/slider-with-input";
import { Button } from "./ui/button";

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
  onTransitionSubmit: (transition: PromptTransition) => void;
  interpolationMethod: "linear" | "slerp";
  onInterpolationMethodChange: (method: "linear" | "slerp") => void;
  temporalInterpolationMethod: "linear" | "slerp";
  onTemporalInterpolationMethodChange: (method: "linear" | "slerp") => void;
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  selectedTimelinePrompt?: TimelinePrompt | null;
  onTimelinePromptUpdate?: (prompt: TimelinePrompt) => void;
  isVideoPaused?: boolean;
  isTimelinePlaying?: boolean;
  currentTime?: number;
  timelinePrompts?: TimelinePrompt[];
  transitionSteps: number;
  onTransitionStepsChange: (steps: number) => void;
  // Spout input settings
  spoutReceiverName?: string;
  onSpoutReceiverChange?: (name: string) => void;
  // Input mode (text vs video) for multi-mode pipelines
  inputMode: InputMode;
  onInputModeChange: (mode: InputMode) => void;
  // Whether Spout is available (server-side detection for native Windows, not WSL)
  spoutAvailable?: boolean;
  // VACE-specific props
  refImages?: string[];
  onRefImagesChange?: (images: string[]) => void;
  vaceContextScale?: number;
  onVaceContextScaleChange?: (scale: number) => void;
  onSendHints?: (imagePaths: string[]) => void;
  isDownloading?: boolean;
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
  onTransitionSubmit,
  interpolationMethod,
  onInterpolationMethodChange,
  temporalInterpolationMethod,
  onTemporalInterpolationMethodChange,
  isLive = false,
  onLivePromptSubmit,
  selectedTimelinePrompt = null,
  onTimelinePromptUpdate,
  isVideoPaused = false,
  isTimelinePlaying: _isTimelinePlaying = false,
  currentTime: _currentTime = 0,
  timelinePrompts: _timelinePrompts = [],
  transitionSteps,
  onTransitionStepsChange,
  spoutReceiverName = "",
  onSpoutReceiverChange,
  inputMode,
  onInputModeChange,
  spoutAvailable = false,
  refImages = [],
  onRefImagesChange,
  vaceContextScale = 1.0,
  onVaceContextScaleChange,
  onSendHints,
  isDownloading = false,
}: InputAndControlsPanelProps) {
  // Helper function to determine if playhead is at the end of timeline
  const isAtEndOfTimeline = () => {
    if (_timelinePrompts.length === 0) return true;

    // Live prompts are always at the end, so the last prompt has the latest endTime
    const lastPrompt = _timelinePrompts[_timelinePrompts.length - 1];

    // Check if current time is at or past the end of the last prompt
    return _currentTime >= lastPrompt.endTime;
  };
  const videoRef = useRef<HTMLVideoElement>(null);

  // Check if this pipeline supports multiple input modes
  const isMultiMode = pipelineIsMultiMode(pipelineId);

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
                Video File
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
              /* Spout Receiver Configuration */
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name:"
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
              /* Video/Camera Input Preview */
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
            )}
          </div>
        )}

        {/* VACE Image Controls */}
        <div className="space-y-4">
          <ImageManager
            images={refImages}
            onImagesChange={onRefImagesChange || (() => {})}
            disabled={isDownloading}
          />

          {refImages && refImages.length > 0 && (
            <div className="space-y-2">
              <LabelWithTooltip
                label="VACE Context Scale"
                tooltip="Scaling factor for VACE hint injection. Higher values make reference images more influential."
                className="text-sm font-medium"
              />
              <SliderWithInput
                value={vaceContextScale}
                onValueChange={onVaceContextScaleChange || (() => {})}
                min={0}
                max={2}
                step={0.1}
                disabled={isDownloading}
              />
              {onSendHints && (
                <div className="flex items-center justify-end">
                  <Button
                    onMouseDown={e => {
                      e.preventDefault();
                      onSendHints(refImages.filter(img => img));
                    }}
                    disabled={isDownloading || !isStreaming}
                    size="sm"
                    className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    title={
                      !isStreaming
                        ? "Start streaming to send hints"
                        : "Submit all reference images"
                    }
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>

        <div>
          {(() => {
            // The Input can have two states: Append (default) and Edit (when a prompt is selected and the video is paused)
            const isEditMode = selectedTimelinePrompt && isVideoPaused;

            return (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">Prompts</h3>
                  {isEditMode && (
                    <Badge variant="secondary" className="text-xs">
                      Editing
                    </Badge>
                  )}
                </div>

                {selectedTimelinePrompt ? (
                  <TimelinePromptEditor
                    prompt={selectedTimelinePrompt}
                    onPromptUpdate={onTimelinePromptUpdate}
                    disabled={false}
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    promptIndex={_timelinePrompts.findIndex(
                      p => p.id === selectedTimelinePrompt.id
                    )}
                  />
                ) : (
                  <PromptInput
                    prompts={prompts}
                    onPromptsChange={onPromptsChange}
                    onPromptsSubmit={onPromptsSubmit}
                    onTransitionSubmit={onTransitionSubmit}
                    disabled={
                      pipelineId === "passthrough" ||
                      (_isTimelinePlaying &&
                        !isVideoPaused &&
                        !isAtEndOfTimeline()) ||
                      // Disable in Append mode when paused and not at end
                      (!selectedTimelinePrompt &&
                        isVideoPaused &&
                        !isAtEndOfTimeline())
                    }
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    temporalInterpolationMethod={temporalInterpolationMethod}
                    onTemporalInterpolationMethodChange={
                      onTemporalInterpolationMethodChange
                    }
                    isLive={isLive}
                    onLivePromptSubmit={onLivePromptSubmit}
                    isStreaming={isStreaming}
                    transitionSteps={transitionSteps}
                    onTransitionStepsChange={onTransitionStepsChange}
                    timelinePrompts={_timelinePrompts}
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
