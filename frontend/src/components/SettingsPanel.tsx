import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Badge } from "./ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Toggle } from "./ui/toggle";
import { SliderWithInput } from "./ui/slider-with-input";
import { Hammer, Info, Minus, Plus, RotateCcw } from "lucide-react";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../lib/utils";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type {
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  InputMode,
  PipelineInfo,
  VaeType,
} from "../types";
import { LoRAManager } from "./LoRAManager";

// Minimum dimension for most pipelines (will be overridden by pipeline-specific minDimension from schema)
const DEFAULT_MIN_DIMENSION = 1;

interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  // Resolution is required - parent should always provide from schema defaults
  resolution: {
    height: number;
    width: number;
  };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (denoisingSteps: number[]) => void;
  // Default denoising steps for reset functionality - derived from backend schema
  defaultDenoisingSteps: number[];
  noiseScale?: number;
  onNoiseScaleChange?: (noiseScale: number) => void;
  noiseController?: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  manageCache?: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  quantization?: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
  kvCacheAttentionBias?: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;
  onResetCache?: () => void;
  loras?: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  loraMergeStrategy?: LoraMergeStrategy;
  // Input mode for conditional rendering of noise controls
  inputMode?: InputMode;
  // Whether this pipeline supports noise controls in video mode (schema-derived)
  supportsNoiseControls?: boolean;
  // Spout settings
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
  // Whether Spout is available (server-side detection for native Windows, not WSL)
  spoutAvailable?: boolean;
  // VACE settings
  vaceEnabled?: boolean;
  onVaceEnabledChange?: (enabled: boolean) => void;
  vaceUseInputVideo?: boolean;
  onVaceUseInputVideoChange?: (enabled: boolean) => void;
  vaceContextScale?: number;
  onVaceContextScaleChange?: (scale: number) => void;
  // VAE type selection
  vaeType?: VaeType;
  onVaeTypeChange?: (vaeType: VaeType) => void;
  // Available VAE types from backend registry
  vaeTypes?: string[];
  // Preprocessors
  preprocessorIds?: string[];
  onPreprocessorIdsChange?: (ids: string[]) => void;
  // Postprocessors
  postprocessorIds?: string[];
  onPostprocessorIdsChange?: (ids: string[]) => void;
}

export function SettingsPanel({
  className = "",
  pipelines,
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  isLoading = false,
  resolution,
  onResolutionChange,
  seed = 42,
  onSeedChange,
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
  defaultDenoisingSteps,
  noiseScale = 0.7,
  onNoiseScaleChange,
  noiseController = true,
  onNoiseControllerChange,
  manageCache = true,
  onManageCacheChange,
  quantization = "fp8_e4m3fn",
  onQuantizationChange,
  kvCacheAttentionBias = 0.3,
  onKvCacheAttentionBiasChange,
  onResetCache,
  loras = [],
  onLorasChange,
  loraMergeStrategy = "permanent_merge",
  inputMode,
  supportsNoiseControls = false,
  spoutSender,
  onSpoutSenderChange,
  spoutAvailable = false,
  vaceEnabled = true,
  onVaceEnabledChange,
  vaceUseInputVideo = true,
  onVaceUseInputVideoChange,
  vaceContextScale = 1.0,
  onVaceContextScaleChange,
  vaeType = "wan",
  onVaeTypeChange,
  vaeTypes,
  preprocessorIds = [],
  onPreprocessorIdsChange,
  postprocessorIds = [],
  onPostprocessorIdsChange,
}: SettingsPanelProps) {
  // Local slider state management hooks
  const noiseScaleSlider = useLocalSliderValue(noiseScale, onNoiseScaleChange);
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    onKvCacheAttentionBiasChange
  );
  const vaceContextScaleSlider = useLocalSliderValue(
    vaceContextScale,
    onVaceContextScaleChange
  );

  // Validation error states
  const [heightError, setHeightError] = useState<string | null>(null);
  const [widthError, setWidthError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  // Check if resolution needs adjustment
  const scaleFactor = getResolutionScaleFactor(pipelineId);
  const resolutionWarning =
    scaleFactor &&
    (resolution.height % scaleFactor !== 0 ||
      resolution.width % scaleFactor !== 0)
      ? `Resolution will be adjusted to ${adjustResolutionForPipeline(pipelineId, resolution).resolution.width}×${adjustResolutionForPipeline(pipelineId, resolution).resolution.height} when starting the stream (must be divisible by ${scaleFactor})`
      : null;

  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    // Get min dimension from pipeline schema, fallback to default
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const maxValue = 2048;

    // Validate and set error state
    if (value < minValue) {
      if (dimension === "height") {
        setHeightError(`Must be at least ${minValue}`);
      } else {
        setWidthError(`Must be at least ${minValue}`);
      }
    } else if (value > maxValue) {
      if (dimension === "height") {
        setHeightError(`Must be at most ${maxValue}`);
      } else {
        setWidthError(`Must be at most ${maxValue}`);
      }
    } else {
      // Clear error if valid
      if (dimension === "height") {
        setHeightError(null);
      } else {
        setWidthError(null);
      }
    }

    // Always update the value (even if invalid)
    onResolutionChange?.({
      ...resolution,
      [dimension]: value,
    });
  };

  const incrementResolution = (dimension: "height" | "width") => {
    const maxValue = 2048;
    const newValue = Math.min(maxValue, resolution[dimension] + 1);
    handleResolutionChange(dimension, newValue);
  };

  const decrementResolution = (dimension: "height" | "width") => {
    // Get min dimension from pipeline schema, fallback to default
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const newValue = Math.max(minValue, resolution[dimension] - 1);
    handleResolutionChange(dimension, newValue);
  };

  const handleSeedChange = (value: number) => {
    const minValue = 0;
    const maxValue = 2147483647;

    // Validate and set error state
    if (value < minValue) {
      setSeedError(`Must be at least ${minValue}`);
    } else if (value > maxValue) {
      setSeedError(`Must be at most ${maxValue}`);
    } else {
      setSeedError(null);
    }

    // Always update the value (even if invalid)
    onSeedChange?.(value);
  };

  const incrementSeed = () => {
    const maxValue = 2147483647;
    const newValue = Math.min(maxValue, seed + 1);
    handleSeedChange(newValue);
  };

  const decrementSeed = () => {
    const minValue = 0;
    const newValue = Math.max(minValue, seed - 1);
    handleSeedChange(newValue);
  };

  const currentPipeline = pipelines?.[pipelineId];

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Pipeline ID</h3>
          <Select
            value={pipelineId}
            onValueChange={handlePipelineIdChange}
            disabled={isStreaming || isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a pipeline" />
            </SelectTrigger>
            <SelectContent>
              {pipelines &&
                Object.entries(pipelines).map(([id]) => (
                  <SelectItem key={id} value={id}>
                    {id}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>

        {currentPipeline && (
          <Card>
            <CardContent className="p-4 space-y-2">
              <div>
                <h4 className="text-sm font-semibold">
                  {currentPipeline.name}
                </h4>
              </div>

              <div>
                {(currentPipeline.about ||
                  currentPipeline.docsUrl ||
                  currentPipeline.modified) && (
                  <div className="flex items-stretch gap-1 h-6">
                    {currentPipeline.about && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge
                              variant="outline"
                              className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                            >
                              <Info className="h-3.5 w-3.5" />
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">{currentPipeline.about}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    {currentPipeline.modified && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge
                              variant="outline"
                              className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                            >
                              <Hammer className="h-3.5 w-3.5" />
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>
                              This pipeline contains modifications based on the
                              original project.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    {currentPipeline.docsUrl && (
                      <a
                        href={currentPipeline.docsUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block h-full"
                      >
                        <Badge
                          variant="outline"
                          className="hover:bg-accent cursor-pointer h-full flex items-center"
                        >
                          Docs
                        </Badge>
                      </a>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* VACE Toggle */}
        {currentPipeline?.supportsVACE && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="VACE"
                tooltip="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, you can use reference images for R2V generation. In Video input mode, a separate toggle controls whether the input video is used for VACE conditioning or for latent initialization. Requires pipeline reload to take effect."
                className="text-sm font-medium"
              />
              <Toggle
                pressed={vaceEnabled}
                onPressedChange={onVaceEnabledChange || (() => {})}
                variant="outline"
                size="sm"
                className="h-7"
                disabled={isStreaming || isLoading}
              >
                {vaceEnabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {/* Warning when VACE is enabled and quantization is set */}
            {vaceEnabled && quantization !== null && (
              <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
                <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
                <p className="text-xs text-amber-600 dark:text-amber-500">
                  VACE is incompatible with FP8 quantization. Please disable
                  quantization to use VACE.
                </p>
              </div>
            )}

            {vaceEnabled && (
              <div className="rounded-lg border bg-card p-3 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label="Use Input Video"
                    tooltip="When enabled in Video input mode, the input video is used for VACE conditioning. When disabled, the input video is used for latent initialization instead, allowing you to use reference images while in Video input mode."
                    className="text-xs text-muted-foreground"
                  />
                  <Toggle
                    pressed={vaceUseInputVideo}
                    onPressedChange={onVaceUseInputVideoChange || (() => {})}
                    variant="outline"
                    size="sm"
                    className="h-7"
                    disabled={isStreaming || isLoading || inputMode !== "video"}
                  >
                    {vaceUseInputVideo ? "ON" : "OFF"}
                  </Toggle>
                </div>
                <div className="flex items-center gap-2">
                  <LabelWithTooltip
                    label="Scale:"
                    tooltip="Scaling factor for VACE hint injection. Higher values make reference images more influential."
                    className="text-xs text-muted-foreground w-16"
                  />
                  <div className="flex-1 min-w-0">
                    <SliderWithInput
                      value={vaceContextScaleSlider.localValue}
                      onValueChange={vaceContextScaleSlider.handleValueChange}
                      onValueCommit={vaceContextScaleSlider.handleValueCommit}
                      min={0}
                      max={2}
                      step={0.1}
                      incrementAmount={0.1}
                      valueFormatter={vaceContextScaleSlider.formatValue}
                      inputParser={v => parseFloat(v) || 1.0}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {currentPipeline?.supportsLoRA && (
          <div className="space-y-4">
            <LoRAManager
              loras={loras}
              onLorasChange={onLorasChange}
              disabled={isLoading}
              isStreaming={isStreaming}
              loraMergeStrategy={loraMergeStrategy}
            />
          </div>
        )}

        {/* Preprocessor Selector */}
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={PARAMETER_METADATA.preprocessor.label}
              tooltip={PARAMETER_METADATA.preprocessor.tooltip}
              className="text-sm text-foreground"
            />
            <Select
              value={preprocessorIds.length > 0 ? preprocessorIds[0] : "none"}
              onValueChange={value => {
                if (value === "none") {
                  onPreprocessorIdsChange?.([]);
                } else {
                  onPreprocessorIdsChange?.([value]);
                }
              }}
              disabled={isStreaming || isLoading}
            >
              <SelectTrigger className="w-[140px] h-7">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {Object.entries(pipelines || {})
                  .filter(([, info]) => {
                    const isPreprocessor =
                      info.usage?.includes("preprocessor") ?? false;
                    if (!isPreprocessor) return false;
                    // Filter by input mode: only show preprocessors that support the current input mode
                    if (inputMode) {
                      return info.supportedModes?.includes(inputMode) ?? false;
                    }
                    return true;
                  })
                  .map(([pid]) => (
                    <SelectItem key={pid} value={pid}>
                      {pid}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Postprocessor Selector */}
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={PARAMETER_METADATA.postprocessor.label}
              tooltip={PARAMETER_METADATA.postprocessor.tooltip}
              className="text-sm text-foreground"
            />
            <Select
              value={postprocessorIds.length > 0 ? postprocessorIds[0] : "none"}
              onValueChange={value => {
                if (value === "none") {
                  onPostprocessorIdsChange?.([]);
                } else {
                  onPostprocessorIdsChange?.([value]);
                }
              }}
              disabled={isStreaming || isLoading}
            >
              <SelectTrigger className="w-[140px] h-7">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {Object.entries(pipelines || {})
                  .filter(([, info]) => {
                    const isPostprocessor =
                      info.usage?.includes("postprocessor") ?? false;
                    if (!isPostprocessor) return false;
                    // Postprocessors run after the main pipeline, so they receive video output.
                    // Show any postprocessor that supports video mode, regardless of current input mode.
                    return info.supportedModes?.includes("video") ?? false;
                  })
                  .map(([pid]) => (
                    <SelectItem key={pid} value={pid}>
                      {pid}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* VAE Type Selection */}
        {vaeTypes && vaeTypes.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.vaeType.label}
                tooltip={PARAMETER_METADATA.vaeType.tooltip}
                className="text-sm text-foreground"
              />
              <Select
                value={vaeType}
                onValueChange={value => {
                  onVaeTypeChange?.(value as VaeType);
                }}
                disabled={isStreaming || isLoading}
              >
                <SelectTrigger className="w-[140px] h-7">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {vaeTypes.map(type => (
                    <SelectItem key={type} value={type}>
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        {/* Resolution controls - shown for pipelines that support quantization (implies they need resolution config) */}
        {pipelines?.[pipelineId]?.supportsQuantization && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <LabelWithTooltip
                      label={PARAMETER_METADATA.height.label}
                      tooltip={PARAMETER_METADATA.height.tooltip}
                      className="text-sm text-foreground w-14"
                    />
                    <div
                      className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${heightError ? "border-red-500" : ""}`}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                        onClick={() => decrementResolution("height")}
                        disabled={isStreaming}
                      >
                        <Minus className="h-3.5 w-3.5" />
                      </Button>
                      <Input
                        type="number"
                        value={resolution.height}
                        onChange={e => {
                          const value = parseInt(e.target.value);
                          if (!isNaN(value)) {
                            handleResolutionChange("height", value);
                          }
                        }}
                        disabled={isStreaming}
                        className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        min={
                          pipelines?.[pipelineId]?.minDimension ??
                          DEFAULT_MIN_DIMENSION
                        }
                        max={2048}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                        onClick={() => incrementResolution("height")}
                        disabled={isStreaming}
                      >
                        <Plus className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  {heightError && (
                    <p className="text-xs text-red-500 ml-16">{heightError}</p>
                  )}
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <LabelWithTooltip
                      label={PARAMETER_METADATA.width.label}
                      tooltip={PARAMETER_METADATA.width.tooltip}
                      className="text-sm text-foreground w-14"
                    />
                    <div
                      className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${widthError ? "border-red-500" : ""}`}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                        onClick={() => decrementResolution("width")}
                        disabled={isStreaming}
                      >
                        <Minus className="h-3.5 w-3.5" />
                      </Button>
                      <Input
                        type="number"
                        value={resolution.width}
                        onChange={e => {
                          const value = parseInt(e.target.value);
                          if (!isNaN(value)) {
                            handleResolutionChange("width", value);
                          }
                        }}
                        disabled={isStreaming}
                        className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        min={
                          pipelines?.[pipelineId]?.minDimension ??
                          DEFAULT_MIN_DIMENSION
                        }
                        max={2048}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                        onClick={() => incrementResolution("width")}
                        disabled={isStreaming}
                      >
                        <Plus className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  {widthError && (
                    <p className="text-xs text-red-500 ml-16">{widthError}</p>
                  )}
                </div>
                {resolutionWarning && (
                  <div className="flex items-start gap-1">
                    <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
                    <p className="text-xs text-amber-600 dark:text-amber-500">
                      {resolutionWarning}
                    </p>
                  </div>
                )}
              </div>

              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.seed.label}
                    tooltip={PARAMETER_METADATA.seed.tooltip}
                    className="text-sm text-foreground w-14"
                  />
                  <div
                    className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${seedError ? "border-red-500" : ""}`}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                      onClick={decrementSeed}
                      disabled={isStreaming}
                    >
                      <Minus className="h-3.5 w-3.5" />
                    </Button>
                    <Input
                      type="number"
                      value={seed}
                      onChange={e => {
                        const value = parseInt(e.target.value);
                        if (!isNaN(value)) {
                          handleSeedChange(value);
                        }
                      }}
                      disabled={isStreaming}
                      className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      min={0}
                      max={2147483647}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                      onClick={incrementSeed}
                      disabled={isStreaming}
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
                {seedError && (
                  <p className="text-xs text-red-500 ml-16">{seedError}</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Cache management controls - shown for pipelines that support it */}
        {pipelines?.[pipelineId]?.supportsCacheManagement && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                {/* KV Cache bias control - shown for pipelines that support it */}
                {pipelines?.[pipelineId]?.supportsKvCacheBias && (
                  <SliderWithInput
                    label={PARAMETER_METADATA.kvCacheAttentionBias.label}
                    tooltip={PARAMETER_METADATA.kvCacheAttentionBias.tooltip}
                    value={kvCacheAttentionBiasSlider.localValue}
                    onValueChange={kvCacheAttentionBiasSlider.handleValueChange}
                    onValueCommit={kvCacheAttentionBiasSlider.handleValueCommit}
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    incrementAmount={0.01}
                    labelClassName="text-sm text-foreground w-20"
                    valueFormatter={kvCacheAttentionBiasSlider.formatValue}
                    inputParser={v => parseFloat(v) || 1.0}
                  />
                )}

                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.manageCache.label}
                    tooltip={PARAMETER_METADATA.manageCache.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Toggle
                    pressed={manageCache}
                    onPressedChange={onManageCacheChange || (() => {})}
                    variant="outline"
                    size="sm"
                    className="h-7"
                  >
                    {manageCache ? "ON" : "OFF"}
                  </Toggle>
                </div>

                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.resetCache.label}
                    tooltip={PARAMETER_METADATA.resetCache.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Button
                    type="button"
                    onClick={onResetCache || (() => {})}
                    disabled={manageCache}
                    variant="outline"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Denoising steps - shown for pipelines that support quantization (implies advanced diffusion features) */}
        {pipelines?.[pipelineId]?.supportsQuantization && (
          <DenoisingStepsSlider
            value={denoisingSteps}
            onChange={onDenoisingStepsChange || (() => {})}
            defaultValues={defaultDenoisingSteps}
            tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
          />
        )}

        {/* Noise controls - show for video mode on supported pipelines (schema-derived) */}
        {inputMode === "video" && supportsNoiseControls && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.noiseController.label}
                    tooltip={PARAMETER_METADATA.noiseController.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Toggle
                    pressed={noiseController}
                    onPressedChange={onNoiseControllerChange || (() => {})}
                    disabled={isStreaming}
                    variant="outline"
                    size="sm"
                    className="h-7"
                  >
                    {noiseController ? "ON" : "OFF"}
                  </Toggle>
                </div>
              </div>

              <SliderWithInput
                label={PARAMETER_METADATA.noiseScale.label}
                tooltip={PARAMETER_METADATA.noiseScale.tooltip}
                value={noiseScaleSlider.localValue}
                onValueChange={noiseScaleSlider.handleValueChange}
                onValueCommit={noiseScaleSlider.handleValueCommit}
                min={0.0}
                max={1.0}
                step={0.01}
                incrementAmount={0.01}
                disabled={noiseController}
                labelClassName="text-sm text-foreground w-20"
                valueFormatter={noiseScaleSlider.formatValue}
                inputParser={v => parseFloat(v) || 0.0}
              />
            </div>
          </div>
        )}

        {/* Quantization controls - shown for pipelines that support it */}
        {pipelines?.[pipelineId]?.supportsQuantization && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.quantization.label}
                    tooltip={PARAMETER_METADATA.quantization.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Select
                    value={quantization || "none"}
                    onValueChange={value => {
                      onQuantizationChange?.(
                        value === "none" ? null : (value as "fp8_e4m3fn")
                      );
                    }}
                    disabled={isStreaming || vaceEnabled}
                  >
                    <SelectTrigger className="w-[140px] h-7">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">None</SelectItem>
                      <SelectItem value="fp8_e4m3fn">
                        fp8_e4m3fn (Dynamic)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {/* Note when quantization is disabled due to VACE */}
                {vaceEnabled && (
                  <p className="text-xs text-muted-foreground">
                    Disabled because VACE is enabled. Disable VACE to use FP8
                    quantization.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Spout Sender Settings (available on native Windows only) */}
        {spoutAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.spoutSender.label}
                tooltip={PARAMETER_METADATA.spoutSender.tooltip}
                className="text-sm text-foreground"
              />
              <Toggle
                pressed={spoutSender?.enabled ?? false}
                onPressedChange={enabled => {
                  onSpoutSenderChange?.({
                    enabled,
                    name: spoutSender?.name ?? "ScopeOut",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {spoutSender?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {spoutSender?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name:"
                  tooltip="The name of the sender that will send video to Spout-compatible apps like TouchDesigner, Resolume, OBS."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={spoutSender?.name ?? "ScopeOut"}
                  onChange={e => {
                    onSpoutSenderChange?.({
                      enabled: spoutSender?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="ScopeOut"
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
