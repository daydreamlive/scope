import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";
import { Switch } from "./ui/switch";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Upload, ArrowUp, RefreshCw } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type {
  PromptItem,
  PromptTransition,
  DiscoveredSource,
} from "../lib/api";
import { getInputSourceSources, getInputSourceStreamUrl } from "../lib/api";
import type { ExtensionMode, InputMode, PipelineInfo } from "../types";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import type { TimelinePrompt } from "./PromptTimeline";
import { ImageManager } from "./ImageManager";
import { Button } from "./ui/button";
import {
  type ConfigSchemaLike,
  type PrimitiveFieldType,
  COMPLEX_COMPONENTS,
  parseInputFields,
} from "../lib/schemaSettings";
import { SchemaPrimitiveField } from "./PrimitiveFields";
import { useCloudStatus } from "../hooks/useCloudStatus";

interface InputAndControlsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
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
  // Whether NDI is available (NDI SDK installed on server)
  ndiAvailable?: boolean;
  // Whether Syphon is available (macOS only)
  syphonAvailable?: boolean;
  // Currently selected NDI source identifier
  selectedNdiSource?: string;
  onNdiSourceChange?: (identifier: string) => void;
  // Currently selected Syphon source identifier
  selectedSyphonSource?: string;
  onSyphonSourceChange?: (identifier: string) => void;
  // VACE reference images (only shown when VACE is enabled)
  vaceEnabled?: boolean;
  refImages?: string[];
  onRefImagesChange?: (images: string[]) => void;
  onSendHints?: (imagePaths: string[]) => void;
  isDownloading?: boolean;
  // Images input support - presence of images field in pipeline schema
  supportsImages?: boolean;
  // FFLF (First-Frame-Last-Frame) extension mode
  firstFrameImage?: string;
  onFirstFrameImageChange?: (imagePath: string | undefined) => void;
  lastFrameImage?: string;
  onLastFrameImageChange?: (imagePath: string | undefined) => void;
  extensionMode?: ExtensionMode;
  onExtensionModeChange?: (mode: ExtensionMode) => void;
  onSendExtensionFrames?: () => void;
  // Schema-driven input fields (category "input"), shown below Prompts
  configSchema?: ConfigSchemaLike;
  schemaFieldOverrides?: Record<string, unknown>;
  onSchemaFieldOverrideChange?: (
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
}

export function InputAndControlsPanel({
  className = "",
  pipelines,
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
  ndiAvailable = false,
  syphonAvailable = false,
  selectedNdiSource = "",
  onNdiSourceChange,
  selectedSyphonSource = "",
  onSyphonSourceChange,
  vaceEnabled = true,
  refImages = [],
  onRefImagesChange,
  onSendHints,
  isDownloading = false,
  supportsImages = false,
  firstFrameImage,
  onFirstFrameImageChange,
  lastFrameImage,
  onLastFrameImageChange,
  extensionMode = "firstframe",
  onExtensionModeChange,
  onSendExtensionFrames,
  configSchema,
  schemaFieldOverrides,
  onSchemaFieldOverrideChange,
}: InputAndControlsPanelProps) {
  // NDI source discovery
  const [ndiSources, setNdiSources] = useState<DiscoveredSource[]>([]);
  const [isDiscoveringNdi, setIsDiscoveringNdi] = useState(false);

  const discoverNdiSources = useCallback(async () => {
    setIsDiscoveringNdi(true);
    try {
      const result = await getInputSourceSources("ndi");
      setNdiSources(result.sources);
    } catch (e) {
      console.error("Failed to discover NDI sources:", e);
      setNdiSources([]);
    } finally {
      setIsDiscoveringNdi(false);
    }
  }, []);

  // Live MJPEG preview URL
  const [showNdiPreview, setShowNdiPreview] = useState(false);
  const ndiStreamUrl =
    mode === "ndi" && selectedNdiSource && showNdiPreview
      ? getInputSourceStreamUrl("ndi", selectedNdiSource)
      : null;
  const [isStreamLoaded, setIsStreamLoaded] = useState(false);

  // Reset loaded state when the stream URL changes
  useEffect(() => {
    setIsStreamLoaded(false);
  }, [ndiStreamUrl]);

  // Syphon source discovery
  const [syphonSources, setSyphonSources] = useState<DiscoveredSource[]>([]);
  const [isDiscoveringSyphon, setIsDiscoveringSyphon] = useState(false);

  const discoverSyphonSources = useCallback(async () => {
    setIsDiscoveringSyphon(true);
    try {
      const result = await getInputSourceSources("syphon");
      setSyphonSources(result.sources);
    } catch (e) {
      console.error("Failed to discover Syphon sources:", e);
      setSyphonSources([]);
    } finally {
      setIsDiscoveringSyphon(false);
    }
  }, []);

  // Live MJPEG preview URL for Syphon (always shown when a source is selected)
  // Use higher FPS for Syphon since it's local GPU sharing with minimal overhead
  const syphonStreamUrl =
    mode === "syphon" && selectedSyphonSource
      ? getInputSourceStreamUrl("syphon", selectedSyphonSource, 15)
      : null;
  const [isSyphonStreamLoaded, setIsSyphonStreamLoaded] = useState(false);

  useEffect(() => {
    setIsSyphonStreamLoaded(false);
  }, [syphonStreamUrl]);

  // Auto-discover Syphon sources when switching to Syphon mode
  useEffect(() => {
    if (mode === "syphon" && syphonAvailable) {
      discoverSyphonSources();
    }
  }, [mode, syphonAvailable, discoverSyphonSources]);

  // Auto-discover NDI sources when switching to NDI mode
  useEffect(() => {
    if (mode === "ndi" && ndiAvailable) {
      discoverNdiSources();
    }
  }, [mode, ndiAvailable, discoverNdiSources]);

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

  // Track cloud connection state and clear images when it changes
  // (switching between local/cloud means different asset lists)
  const { isConnected: isCloudConnected } = useCloudStatus();
  const prevCloudConnectedRef = useRef<boolean | null>(null);

  useEffect(() => {
    // On first render, just store the initial state
    if (prevCloudConnectedRef.current === null) {
      prevCloudConnectedRef.current = isCloudConnected;
      return;
    }

    // Clear images when cloud connection state changes (connected or disconnected)
    if (prevCloudConnectedRef.current !== isCloudConnected) {
      onRefImagesChange?.([]);
      onFirstFrameImageChange?.(undefined);
      onLastFrameImageChange?.(undefined);
    }

    prevCloudConnectedRef.current = isCloudConnected;
  }, [
    isCloudConnected,
    onRefImagesChange,
    onFirstFrameImageChange,
    onLastFrameImageChange,
  ]);

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
              <ToggleGroupItem
                value="video"
                aria-label="Video file"
                disabled={isStreaming && (mode === "ndi" || mode === "syphon")}
              >
                File
              </ToggleGroupItem>
              <ToggleGroupItem
                value="camera"
                aria-label="Camera"
                disabled={isStreaming && (mode === "ndi" || mode === "syphon")}
              >
                Camera
              </ToggleGroupItem>
              {spoutAvailable && (
                <ToggleGroupItem
                  value="spout"
                  aria-label="Spout Receiver"
                  disabled={
                    isStreaming && (mode === "ndi" || mode === "syphon")
                  }
                >
                  Spout
                </ToggleGroupItem>
              )}
              {ndiAvailable && (
                <ToggleGroupItem
                  value="ndi"
                  aria-label="NDI"
                  disabled={isStreaming && mode !== "ndi"}
                >
                  NDI
                </ToggleGroupItem>
              )}
              {syphonAvailable && (
                <ToggleGroupItem
                  value="syphon"
                  aria-label="Syphon"
                  disabled={isStreaming && mode !== "syphon"}
                >
                  Syphon
                </ToggleGroupItem>
              )}
            </ToggleGroup>
          </div>
        )}

        {/* Video preview - only show when in video input mode */}
        {inputMode === "video" && (
          <div>
            <h3 className="text-sm font-medium mb-2">Input</h3>
            {mode === "ndi" ? (
              /* NDI Source Picker */
              <div className="space-y-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Select
                    value={selectedNdiSource}
                    onValueChange={value => onNdiSourceChange?.(value)}
                    disabled={isStreaming || isDiscoveringNdi}
                  >
                    <SelectTrigger className="flex-1 min-w-0 h-8 text-sm [&>span]:truncate">
                      <SelectValue
                        placeholder={
                          isDiscoveringNdi
                            ? "Discovering..."
                            : ndiSources.length === 0
                              ? "No sources found"
                              : "Select NDI source"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {ndiSources.map(source => (
                        <SelectItem
                          key={source.identifier}
                          value={source.identifier}
                        >
                          {source.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={discoverNdiSources}
                    disabled={isStreaming || isDiscoveringNdi}
                    title="Refresh NDI sources"
                  >
                    <RefreshCw
                      className={`h-3.5 w-3.5 ${isDiscoveringNdi ? "animate-spin" : ""}`}
                    />
                  </Button>
                </div>
                {/* Preview toggle */}
                {selectedNdiSource && (
                  <div className="flex items-center gap-2 text-xs">
                    <Switch
                      id="ndi-preview-toggle"
                      checked={showNdiPreview}
                      onCheckedChange={setShowNdiPreview}
                      disabled={isStreaming}
                    />
                    <label
                      htmlFor="ndi-preview-toggle"
                      className="text-muted-foreground cursor-pointer select-none"
                    >
                      Show live preview
                    </label>
                  </div>
                )}
                {/* Live NDI preview (MJPEG stream) */}
                {selectedNdiSource && showNdiPreview && (
                  <div className="relative rounded-md overflow-hidden border border-border bg-muted min-w-0">
                    {ndiStreamUrl ? (
                      <img
                        src={ndiStreamUrl}
                        alt="NDI source preview"
                        className="block w-full h-auto object-contain"
                        onLoad={() => setIsStreamLoaded(true)}
                        onError={() => setIsStreamLoaded(false)}
                      />
                    ) : null}
                    {!isStreamLoaded && (
                      <div className="flex items-center justify-center h-20 text-xs text-muted-foreground">
                        <RefreshCw className="h-4 w-4 animate-spin mr-1.5" />
                        Connecting...
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : mode === "spout" ? (
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
            ) : mode === "syphon" ? (
              /* Syphon Source Picker */
              <div className="space-y-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Select
                    value={selectedSyphonSource}
                    onValueChange={value => onSyphonSourceChange?.(value)}
                    disabled={isStreaming || isDiscoveringSyphon}
                  >
                    <SelectTrigger className="flex-1 min-w-0 h-8 text-sm [&>span]:truncate">
                      <SelectValue
                        placeholder={
                          isDiscoveringSyphon
                            ? "Discovering..."
                            : syphonSources.length === 0
                              ? "No sources found"
                              : "Select Syphon source"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {syphonSources.map(source => (
                        <SelectItem
                          key={source.identifier}
                          value={source.identifier}
                        >
                          {source.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={discoverSyphonSources}
                    disabled={isStreaming || isDiscoveringSyphon}
                    title="Refresh Syphon sources"
                  >
                    <RefreshCw
                      className={`h-3.5 w-3.5 ${isDiscoveringSyphon ? "animate-spin" : ""}`}
                    />
                  </Button>
                </div>
                {/* Live Syphon preview (MJPEG stream) */}
                {selectedSyphonSource && (
                  <div className="relative rounded-md overflow-hidden border border-border bg-muted min-w-0">
                    {syphonStreamUrl ? (
                      <img
                        src={syphonStreamUrl}
                        alt="Syphon source preview"
                        className="block w-full h-auto object-contain"
                        onLoad={() => setIsSyphonStreamLoaded(true)}
                        onError={() => setIsSyphonStreamLoaded(false)}
                      />
                    ) : null}
                    {!isSyphonStreamLoaded && (
                      <div className="flex items-center justify-center h-20 text-xs text-muted-foreground">
                        <RefreshCw className="h-4 w-4 animate-spin mr-1.5" />
                        Connecting...
                      </div>
                    )}
                  </div>
                )}
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
