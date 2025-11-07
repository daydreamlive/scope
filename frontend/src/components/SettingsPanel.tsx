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
import { PIPELINES } from "../data/pipelines";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { getDefaultDenoisingSteps, getDefaultResolution } from "../lib/utils";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type { PipelineId } from "../types";

const MIN_DIMENSION = 16;

interface SettingsPanelProps {
  className?: string;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isDownloading?: boolean;
  resolution?: {
    height: number;
    width: number;
  };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (denoisingSteps: number[]) => void;
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
}

export function SettingsPanel({
  className = "",
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  isDownloading = false,
  resolution,
  onResolutionChange,
  seed = 42,
  onSeedChange,
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
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
}: SettingsPanelProps) {
  // Use pipeline-specific default if resolution is not provided
  const effectiveResolution = resolution || getDefaultResolution(pipelineId);

  // Local slider state management hooks
  const noiseScaleSlider = useLocalSliderValue(noiseScale, onNoiseScaleChange);
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    onKvCacheAttentionBiasChange
  );

  // Validation error states
  const [heightError, setHeightError] = useState<string | null>(null);
  const [widthError, setWidthError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  const handlePipelineIdChange = (value: string) => {
    if (value in PIPELINES) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    const minValue =
      pipelineId === "longlive" ||
      pipelineId === "streamdiffusionv2" ||
      pipelineId === "krea-realtime-video"
        ? MIN_DIMENSION
        : 1;
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
      ...effectiveResolution,
      [dimension]: value,
    });
  };

  const incrementResolution = (dimension: "height" | "width") => {
    const maxValue = 2048;
    const newValue = Math.min(maxValue, effectiveResolution[dimension] + 1);
    handleResolutionChange(dimension, newValue);
  };

  const decrementResolution = (dimension: "height" | "width") => {
    const minValue =
      pipelineId === "longlive" ||
      pipelineId === "streamdiffusionv2" ||
      pipelineId === "krea-realtime-video"
        ? MIN_DIMENSION
        : 1;
    const newValue = Math.max(minValue, effectiveResolution[dimension] - 1);
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

  const currentPipeline = PIPELINES[pipelineId];

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
            disabled={isStreaming || isDownloading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a pipeline" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(PIPELINES).map(id => (
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

        {(pipelineId === "longlive" ||
          pipelineId === "streamdiffusionv2" ||
          pipelineId === "krea-realtime-video") && (
          <div className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Parameters</h3>

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
                        value={effectiveResolution.height}
                        onChange={e => {
                          const value = parseInt(e.target.value);
                          if (!isNaN(value)) {
                            handleResolutionChange("height", value);
                          }
                        }}
                        disabled={isStreaming}
                        className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        min={MIN_DIMENSION}
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
                        value={effectiveResolution.width}
                        onChange={e => {
                          const value = parseInt(e.target.value);
                          if (!isNaN(value)) {
                            handleResolutionChange("width", value);
                          }
                        }}
                        disabled={isStreaming}
                        className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        min={MIN_DIMENSION}
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
          </div>
        )}

        {(pipelineId === "longlive" ||
          pipelineId === "streamdiffusionv2" ||
          pipelineId === "krea-realtime-video") && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                {pipelineId === "krea-realtime-video" && (
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

        {(pipelineId === "longlive" ||
          pipelineId === "streamdiffusionv2" ||
          pipelineId === "krea-realtime-video") && (
          <DenoisingStepsSlider
            value={denoisingSteps}
            onChange={onDenoisingStepsChange || (() => {})}
            defaultValues={getDefaultDenoisingSteps(pipelineId)}
            tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
          />
        )}

        {pipelineId === "streamdiffusionv2" && (
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

        {pipelineId === "krea-realtime-video" && (
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
                    disabled={isStreaming}
                  >
                    <SelectTrigger className="w-[140px] h-7">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">None</SelectItem>
                      <SelectItem value="fp8_e4m3fn">fp8_e4m3fn</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
