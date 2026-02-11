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
import type { ExtensionMode, InputMode } from "../types";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import { ImageManager } from "./ImageManager";
import { Button } from "./ui/button";
import {
  type ConfigSchemaLike,
  type PrimitiveFieldType,
  COMPLEX_COMPONENTS,
  parseInputFields,
} from "../lib/schemaSettings";
import { SchemaPrimitiveField } from "./PrimitiveFields";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamContext } from "../contexts/StreamContext";

interface InputAndControlsPanelProps {
  className?: string;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  mode: VideoSourceMode;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
}

export function InputAndControlsPanel({
  className = "",
  localStream,
  isInitializing,
  error,
  mode,
  onVideoFileUpload,
}: InputAndControlsPanelProps) {
  const {
    settings,
    prompts,
    onPromptsChange,
    interpolationMethod,
    onInterpolationMethodChange,
    temporalInterpolationMethod,
    onTemporalInterpolationMethodChange,
    transitionSteps,
    onTransitionStepsChange,
    isLive,
    selectedTimelinePrompt,
    _isTimelinePlaying,
    _currentTime,
    _timelinePrompts,
    isDownloading,
  } = useAppStore(
    useShallow(s => ({
      settings: s.settings,
      prompts: s.promptItems,
      onPromptsChange: s.setPromptItems,
      interpolationMethod: s.interpolationMethod,
      onInterpolationMethodChange: s.setInterpolationMethod,
      temporalInterpolationMethod: s.temporalInterpolationMethod,
      onTemporalInterpolationMethodChange: s.setTemporalInterpolationMethod,
      transitionSteps: s.transitionSteps,
      onTransitionStepsChange: s.setTransitionSteps,
      isLive: s.isLive,
      selectedTimelinePrompt: s.selectedTimelinePrompt,
      _isTimelinePlaying: s.isTimelinePlaying,
      _currentTime: s.timelineCurrentTime,
      _timelinePrompts: s.timelinePrompts,
      isDownloading: s.isDownloading,
    }))
  );

  // Read from contexts
  const { pipelines } = usePipelinesContext();
  const {
    actions,
    isStreaming,
    isConnecting,
    isPipelineLoading: _isPipelineLoading,
    spoutAvailable,
  } = useStreamContext();

  // Derive values from settings/pipelines
  const pipelineId = settings.pipelineId;
  const inputMode = settings.inputMode || "text";
  const isVideoPaused = settings.paused ?? false;
  const spoutReceiverName = settings.spoutReceiver?.name ?? "";
  const vaceEnabled =
    settings.vaceEnabled ??
    (pipelines?.[pipelineId]?.supportsVACE && inputMode !== "video");
  const refImages = settings.refImages || [];
  const supportsImages = pipelines?.[pipelineId]?.supportsImages ?? false;
  const firstFrameImage = settings.firstFrameImage;
  const lastFrameImage = settings.lastFrameImage;
  const extensionMode = settings.extensionMode || "firstframe";
  const configSchema = pipelines?.[pipelineId]?.configSchema as
    | ConfigSchemaLike
    | undefined;
  const schemaFieldOverrides = settings.schemaFieldOverrides ?? {};

  // Alias action handlers to match existing variable names
  const onModeChange = actions.handleModeChange;
  const onPromptsSubmit = onPromptsChange;
  const onTransitionSubmit = actions.handleTransitionSubmit;
  const onLivePromptSubmit = actions.handleLivePromptSubmit;
  const onTimelinePromptUpdate = actions.handleTimelinePromptUpdate;
  const onSpoutReceiverChange = actions.handleSpoutReceiverChange;
  const onInputModeChange = actions.handleInputModeChange;
  const onRefImagesChange = actions.handleRefImagesChange;
  const onSendHints = actions.handleSendHints;
  const onFirstFrameImageChange = actions.handleFirstFrameImageChange;
  const onLastFrameImageChange = actions.handleLastFrameImageChange;
  const onExtensionModeChange = actions.handleExtensionModeChange;
  const onSendExtensionFrames = actions.handleSendExtensionFrames;
  const onSchemaFieldOverrideChange = actions.handleSchemaFieldOverrideChange;
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
  const pipeline = pipelines?.[pipelineId];
  const isMultiMode = (pipeline?.supportedModes?.length ?? 0) > 1;

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
              /* Spout Receiver Configuration */
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

        {/* Reference Images - show when VACE enabled OR when pipeline supports images without VACE */}
        {(vaceEnabled || (supportsImages && !pipeline?.supportsVACE)) && (
          <div>
            <ImageManager
              images={refImages}
              onImagesChange={onRefImagesChange || (() => {})}
              disabled={isDownloading}
              maxImages={1}
              singleColumn={false}
              label={
                vaceEnabled && pipeline?.supportsVACE
                  ? "Reference Images"
                  : "Images"
              }
              tooltip={
                vaceEnabled && pipeline?.supportsVACE
                  ? "Select reference images for VACE conditioning. Images will guide the video generation style and content."
                  : "Select images to send to the pipeline for conditioning."
              }
            />
            {onSendHints && refImages && refImages.length > 0 && (
              <div className="flex items-center justify-end mt-2">
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

        {/* FFLF Extension Frames - only show when VACE is enabled */}
        {vaceEnabled && (
          <div>
            <LabelWithTooltip
              label="Extension Frames"
              tooltip="Set reference frames for video extension. First frame starts the video from that image, last frame generates toward that target."
              className="text-sm font-medium mb-2 block"
            />
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">
                  First Frame
                </span>
                <ImageManager
                  images={firstFrameImage ? [firstFrameImage] : []}
                  onImagesChange={images => {
                    onFirstFrameImageChange?.(images[0] || undefined);
                  }}
                  disabled={isDownloading}
                  maxImages={1}
                  label="First Frame"
                  hideLabel
                />
              </div>
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">
                  Last Frame
                </span>
                <ImageManager
                  images={lastFrameImage ? [lastFrameImage] : []}
                  onImagesChange={images => {
                    onLastFrameImageChange?.(images[0] || undefined);
                  }}
                  disabled={isDownloading}
                  maxImages={1}
                  label="Last Frame"
                  hideLabel
                />
              </div>
            </div>
            {(firstFrameImage || lastFrameImage) && (
              <div className="space-y-2 mt-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground">Mode:</span>
                  <Select
                    value={extensionMode}
                    onValueChange={value => {
                      if (value && onExtensionModeChange) {
                        onExtensionModeChange(value as ExtensionMode);
                      }
                    }}
                    disabled={!firstFrameImage && !lastFrameImage}
                  >
                    <SelectTrigger className="w-24 h-6 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {firstFrameImage && (
                        <SelectItem value="firstframe">First</SelectItem>
                      )}
                      {lastFrameImage && (
                        <SelectItem value="lastframe">Last</SelectItem>
                      )}
                      {firstFrameImage && lastFrameImage && (
                        <SelectItem value="firstlastframe">Both</SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-end">
                  <Button
                    onMouseDown={e => {
                      e.preventDefault();
                      onSendExtensionFrames?.();
                    }}
                    disabled={
                      isDownloading ||
                      !isStreaming ||
                      (!firstFrameImage && !lastFrameImage)
                    }
                    size="sm"
                    className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    title={
                      !isStreaming
                        ? "Start streaming to send extension frames"
                        : "Send extension frames"
                    }
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        <div>
          {(() => {
            // The Input can have two states: Append (default) and Edit (when a prompt is selected and the video is paused)
            const isEditMode = selectedTimelinePrompt && isVideoPaused;

            // Hide prompts section if pipeline doesn't support prompts
            if (pipeline?.supportsPrompts === false) {
              return null;
            }

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
                    defaultTemporalInterpolationMethod={
                      pipeline?.defaultTemporalInterpolationMethod
                    }
                    defaultSpatialInterpolationMethod={
                      pipeline?.defaultSpatialInterpolationMethod
                    }
                  />
                ) : (
                  <PromptInput
                    prompts={prompts}
                    onPromptsChange={onPromptsChange}
                    onPromptsSubmit={onPromptsSubmit}
                    onTransitionSubmit={onTransitionSubmit}
                    disabled={
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
                    defaultTemporalInterpolationMethod={
                      pipeline?.defaultTemporalInterpolationMethod
                    }
                    defaultSpatialInterpolationMethod={
                      pipeline?.defaultSpatialInterpolationMethod
                    }
                  />
                )}
              </div>
            );
          })()}
        </div>

        {/* Schema-driven input fields (category "input"), below app-defined sections */}
        {configSchema &&
          (() => {
            const parsedInputFields = parseInputFields(configSchema, inputMode);
            if (parsedInputFields.length === 0) return null;
            const enumValuesByRef: Record<string, string[]> = {};
            if (configSchema?.$defs) {
              for (const [defName, def] of Object.entries(
                configSchema.$defs as Record<string, { enum?: unknown[] }>
              )) {
                if (def?.enum && Array.isArray(def.enum)) {
                  enumValuesByRef[defName] = def.enum as string[];
                }
              }
            }
            return (
              <div className="space-y-2">
                {parsedInputFields.map(({ key, prop, ui, fieldType }) => {
                  const comp = ui.component;
                  const isRuntimeParam = ui.is_load_param === false;
                  const disabled =
                    (isStreaming && !isRuntimeParam) || _isPipelineLoading;
                  const value = schemaFieldOverrides?.[key] ?? prop.default;
                  const setValue = (v: unknown) =>
                    onSchemaFieldOverrideChange?.(key, v, isRuntimeParam);
                  if (comp === "image") {
                    const path = value == null ? null : String(value);
                    return (
                      <div key={key} className="space-y-1">
                        {ui.label != null && (
                          <span className="text-xs text-muted-foreground">
                            {ui.label}
                          </span>
                        )}
                        <ImageManager
                          images={path ? [path] : []}
                          onImagesChange={images =>
                            onSchemaFieldOverrideChange?.(
                              key,
                              images[0] ?? null,
                              isRuntimeParam
                            )
                          }
                          disabled={disabled}
                          maxImages={1}
                          hideLabel
                        />
                      </div>
                    );
                  }
                  if (
                    comp &&
                    (COMPLEX_COMPONENTS as readonly string[]).includes(comp)
                  ) {
                    return null;
                  }
                  const enumValues =
                    fieldType === "enum" && typeof prop.$ref === "string"
                      ? enumValuesByRef[prop.$ref.split("/").pop() ?? ""]
                      : undefined;
                  const primitiveType: PrimitiveFieldType | undefined =
                    typeof fieldType === "string" &&
                    !(COMPLEX_COMPONENTS as readonly string[]).includes(
                      fieldType
                    )
                      ? (fieldType as PrimitiveFieldType)
                      : undefined;
                  return (
                    <SchemaPrimitiveField
                      key={key}
                      fieldKey={key}
                      prop={prop}
                      value={value}
                      onChange={setValue}
                      disabled={disabled}
                      label={ui.label}
                      fieldType={primitiveType}
                      enumValues={enumValues}
                    />
                  );
                })}
              </div>
            );
          })()}
      </CardContent>
    </Card>
  );
}
