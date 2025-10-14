import { useState, useEffect } from "react";
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
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Toggle } from "./ui/toggle";
import { SliderWithInput } from "./ui/slider-with-input";
import { Hammer, Info, Minus, Plus } from "lucide-react";
import { PIPELINES } from "../data/pipelines";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { getDefaultDenoisingSteps } from "../lib/utils";
import type { PipelineId } from "../types";

const MIN_DIMENSION_LONGLIVE = 16;

interface SettingsPanelProps {
  className?: string;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
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
}

export function SettingsPanel({
  className = "",
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  resolution = { height: 320, width: 576 },
  onResolutionChange,
  seed = 42,
  onSeedChange,
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
  noiseScale = 0.7,
  onNoiseScaleChange,
  noiseController = true,
  onNoiseControllerChange,
}: SettingsPanelProps) {
  // Local state for noise scale for immediate UI feedback
  const [localNoiseScale, setLocalNoiseScale] = useState<number>(noiseScale);

  // Sync with external value changes
  useEffect(() => {
    setLocalNoiseScale(noiseScale);
  }, [noiseScale]);

  const handlePipelineIdChange = (value: string) => {
    if (value in PIPELINES) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    const minValue = pipelineId === "longlive" ? MIN_DIMENSION_LONGLIVE : 1;
    const maxValue = 2048;

    onResolutionChange?.({
      ...resolution,
      [dimension]: Math.max(minValue, Math.min(maxValue, value)),
    });
  };

  const incrementResolution = (dimension: "height" | "width") => {
    const maxValue = 2048;
    const newValue = Math.min(maxValue, resolution[dimension] + 1);
    handleResolutionChange(dimension, newValue);
  };

  const decrementResolution = (dimension: "height" | "width") => {
    const minValue = pipelineId === "longlive" ? MIN_DIMENSION_LONGLIVE : 1;
    const newValue = Math.max(minValue, resolution[dimension] - 1);
    handleResolutionChange(dimension, newValue);
  };

  const handleSeedChange = (value: number) => {
    const minValue = 0;
    const maxValue = 2147483647; // Max 32-bit signed integer
    onSeedChange?.(Math.max(minValue, Math.min(maxValue, value)));
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

  const handleNoiseScaleValueChange = (value: number) => {
    setLocalNoiseScale(value);
  };

  const handleNoiseScaleCommit = (value: number) => {
    onNoiseScaleChange?.(value);
  };

  // Format value to 2 decimal places
  const formatNoiseScale = (value: number) => {
    return Math.round(value * 100) / 100;
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
            disabled={isStreaming}
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
                  currentPipeline.projectUrl ||
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
                    {currentPipeline.projectUrl && (
                      <a
                        href={currentPipeline.projectUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block h-full"
                      >
                        <Badge
                          variant="outline"
                          className="hover:bg-accent cursor-pointer h-full flex items-center"
                        >
                          Project Page
                        </Badge>
                      </a>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {pipelineId === "longlive" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Parameters</h3>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-sm text-foreground w-14">
                    Height:
                  </label>
                  <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8">
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
                      onChange={e =>
                        handleResolutionChange(
                          "height",
                          parseInt(e.target.value) ||
                            (pipelineId === "longlive"
                              ? MIN_DIMENSION_LONGLIVE
                              : 1)
                        )
                      }
                      disabled={isStreaming}
                      className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      min={
                        pipelineId === "longlive" ? MIN_DIMENSION_LONGLIVE : 1
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

                <div className="flex items-center gap-2">
                  <label className="text-sm text-foreground w-14">Width:</label>
                  <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8">
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
                      onChange={e =>
                        handleResolutionChange(
                          "width",
                          parseInt(e.target.value) ||
                            (pipelineId === "longlive"
                              ? MIN_DIMENSION_LONGLIVE
                              : 1)
                        )
                      }
                      disabled={isStreaming}
                      className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      min={
                        pipelineId === "longlive" ? MIN_DIMENSION_LONGLIVE : 1
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

                <div className="flex items-center gap-2">
                  <label className="text-sm text-foreground w-14">Seed:</label>
                  <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8">
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
                      onChange={e =>
                        handleSeedChange(parseInt(e.target.value) || 0)
                      }
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
              </div>
            </div>
          </div>
        )}

        {pipelineId === "streamdiffusionv2" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Parameters</h3>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-sm text-foreground w-14">Seed:</label>
                  <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8">
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
                      onChange={e =>
                        handleSeedChange(parseInt(e.target.value) || 0)
                      }
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
              </div>
            </div>
          </div>
        )}

        {(pipelineId === "longlive" || pipelineId === "streamdiffusionv2") && (
          <DenoisingStepsSlider
            value={denoisingSteps}
            onChange={onDenoisingStepsChange || (() => {})}
            defaultValues={getDefaultDenoisingSteps(pipelineId)}
          />
        )}

        {pipelineId === "streamdiffusionv2" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-foreground">
                      Noise Controller:
                    </label>
                  </div>
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
                label="Noise Scale:"
                value={localNoiseScale}
                onValueChange={handleNoiseScaleValueChange}
                onValueCommit={handleNoiseScaleCommit}
                min={0.0}
                max={1.0}
                step={0.01}
                incrementAmount={0.01}
                disabled={noiseController}
                labelClassName="text-sm text-foreground w-20"
                valueFormatter={formatNoiseScale}
                inputParser={v => parseFloat(v) || 0.0}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
