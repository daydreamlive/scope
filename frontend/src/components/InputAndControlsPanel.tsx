import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Badge } from "./ui/badge";
import { Upload, X } from "lucide-react";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type { PromptItem, PromptTransition } from "../lib/api";
import { PIPELINES } from "../data/pipelines";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import type { TimelinePrompt } from "./PromptTimeline";
import { Button } from "./ui/button";
import { SliderWithInput } from "./ui/slider-with-input";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";

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
  onImageFileUpload?: (file: File) => Promise<boolean>; // New prop for image upload
  onImageClear?: () => void; // New prop for clearing image
  uploadedImage?: string | null; // New prop for displaying uploaded image (base64 or URL)
  clipConditioningScale?: number; // Image conditioning strength
  onClipConditioningScaleChange?: (scale: number) => void; // Handler for scale changes
  i2vMode?: "clip_only" | "channel_concat" | "full"; // I2V conditioning mode
  onI2vModeChange?: (mode: "clip_only" | "channel_concat" | "full") => void; // Handler for mode changes
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
  onImageFileUpload,
  onImageClear,
  uploadedImage,
  clipConditioningScale = 0.5,
  onClipConditioningScaleChange,
  i2vMode = "clip_only",
  onI2vModeChange,
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
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  // Local slider state management for image conditioning scale
  const clipConditioningScaleSlider = useLocalSliderValue(
    clipConditioningScale,
    onClipConditioningScaleChange
  );

  // Get pipeline category, default to video-input
  const pipelineCategory = PIPELINES[pipelineId]?.category || "video-input";

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  // Update image preview when uploadedImage prop changes
  useEffect(() => {
    if (uploadedImage) {
      setImagePreview(uploadedImage);
    }
  }, [uploadedImage]);

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

  const handleImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file && onImageFileUpload) {
      try {
        // Create preview
        const reader = new FileReader();
        reader.onload = e => {
          setImagePreview(e.target?.result as string);
        };
        reader.readAsDataURL(file);

        // Upload the file
        await onImageFileUpload(file);
      } catch (error) {
        console.error("Image upload failed:", error);
      }
    }
    // Reset the input value so the same file can be selected again
    event.target.value = "";
  };

  const handleClearImage = () => {
    setImagePreview(null);
    if (onImageClear) {
      onImageClear();
    }
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
            <h3 className="text-sm font-medium mb-2">Video Input</h3>
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

        {/* Optional Image Input - Available for all modes */}
        <div>
          <h3 className="text-sm font-medium mb-2">Image Input (Optional)</h3>
          <div className="rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative min-h-[120px]">
            {imagePreview ? (
              <div className="relative w-full">
                <img
                  src={imagePreview}
                  alt="Input image preview"
                  className="w-full h-full object-cover"
                />
                {/* Clear button */}
                <Button
                  onClick={handleClearImage}
                  size="sm"
                  variant="ghost"
                  className="absolute top-2 right-2 p-1 rounded-full bg-black/50 hover:bg-black/70"
                >
                  <X className="h-4 w-4 text-white" />
                </Button>
              </div>
            ) : (
              <div className="text-center text-muted-foreground text-sm p-4">
                No image uploaded
              </div>
            )}
            {/* Upload button */}
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="absolute bottom-2 right-2 p-2 rounded-full bg-black/50 transition-colors hover:bg-black/70 cursor-pointer"
            >
              <Upload className="h-4 w-4 text-white" />
            </label>
          </div>

          {/* Image Strength Slider - only show when image is uploaded */}
          {imagePreview && (
            <div className="mt-3 space-y-3">
              <SliderWithInput
                label="Image Strength"
                tooltip="Controls how much the image influences generation. 0.0 = text-only, 0.5 = balanced, 1.0 = maximum image influence. Lower values allow more creative text-driven variations."
                value={clipConditioningScaleSlider.localValue}
                onValueChange={clipConditioningScaleSlider.handleValueChange}
                onValueCommit={clipConditioningScaleSlider.handleValueCommit}
                min={0.0}
                max={1.0}
                step={0.01}
                incrementAmount={0.01}
                labelClassName="text-sm text-foreground w-24"
                valueFormatter={clipConditioningScaleSlider.formatValue}
                inputParser={v => parseFloat(v) || 0.5}
              />

              {/* I2V Mode Selector */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-foreground">I2V Mode</label>
                <Select
                  value={i2vMode}
                  onValueChange={onI2vModeChange}
                >
                  <SelectTrigger className="w-[180px] h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="clip_only">
                      <div className="flex flex-col">
                        <span>CLIP Only</span>
                        <span className="text-xs text-muted-foreground">Semantic guidance</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="channel_concat">
                      <div className="flex flex-col">
                        <span>Channel Concat</span>
                        <span className="text-xs text-muted-foreground">Structural guidance</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="full">
                      <div className="flex flex-col">
                        <span>Full (Both)</span>
                        <span className="text-xs text-muted-foreground">Maximum fidelity</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
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
                      pipelineId === "vod" ||
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
