import { useState } from "react";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Info, Minus, Plus } from "lucide-react";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../../lib/utils";

// Minimum dimension for most pipelines (will be overridden by pipeline-specific minDimension from schema)
const DEFAULT_MIN_DIMENSION = 1;

interface ResolutionControlsProps {
  pipelineId: string;
  resolution: { height: number; width: number };
  minDimension?: number;
  isStreaming: boolean;
  onChange: (dimension: "height" | "width", value: number) => void;
}

export function ResolutionControls({
  pipelineId,
  resolution,
  minDimension = DEFAULT_MIN_DIMENSION,
  isStreaming,
  onChange,
}: ResolutionControlsProps) {
  const [heightError, setHeightError] = useState<string | null>(null);
  const [widthError, setWidthError] = useState<string | null>(null);

  const scaleFactor = getResolutionScaleFactor(pipelineId);
  const resolutionWarning =
    scaleFactor &&
    (resolution.height % scaleFactor !== 0 ||
      resolution.width % scaleFactor !== 0)
      ? `Resolution will be adjusted to ${adjustResolutionForPipeline(pipelineId, resolution).resolution.width}×${adjustResolutionForPipeline(pipelineId, resolution).resolution.height} when starting the stream (must be divisible by ${scaleFactor})`
      : null;

  const maxValue = 2048;

  const handleChange = (dimension: "height" | "width", value: number) => {
    const setError = dimension === "height" ? setHeightError : setWidthError;

    if (value < minDimension) {
      setError(`Must be at least ${minDimension}`);
    } else if (value > maxValue) {
      setError(`Must be at most ${maxValue}`);
    } else {
      setError(null);
    }

    onChange(dimension, value);
  };

  const increment = (dimension: "height" | "width") => {
    handleChange(dimension, Math.min(maxValue, resolution[dimension] + 1));
  };

  const decrement = (dimension: "height" | "width") => {
    handleChange(
      dimension,
      Math.max(minDimension, resolution[dimension] - 1)
    );
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="space-y-2">
          <DimensionInput
            dimension="height"
            value={resolution.height}
            error={heightError}
            minValue={minDimension}
            maxValue={maxValue}
            isStreaming={isStreaming}
            onChange={handleChange}
            onIncrement={increment}
            onDecrement={decrement}
          />

          <DimensionInput
            dimension="width"
            value={resolution.width}
            error={widthError}
            minValue={minDimension}
            maxValue={maxValue}
            isStreaming={isStreaming}
            onChange={handleChange}
            onIncrement={increment}
            onDecrement={decrement}
          />

          {resolutionWarning && (
            <div className="flex items-start gap-1">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
              <p className="text-xs text-amber-600 dark:text-amber-500">
                {resolutionWarning}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface DimensionInputProps {
  dimension: "height" | "width";
  value: number;
  error: string | null;
  minValue: number;
  maxValue: number;
  isStreaming: boolean;
  onChange: (dimension: "height" | "width", value: number) => void;
  onIncrement: (dimension: "height" | "width") => void;
  onDecrement: (dimension: "height" | "width") => void;
}

function DimensionInput({
  dimension,
  value,
  error,
  minValue,
  maxValue,
  isStreaming,
  onChange,
  onIncrement,
  onDecrement,
}: DimensionInputProps) {
  const metadata =
    PARAMETER_METADATA[dimension as keyof typeof PARAMETER_METADATA];

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <LabelWithTooltip
          label={metadata.label}
          tooltip={metadata.tooltip}
          className="text-sm font-medium w-14"
        />
        <div
          className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${error ? "border-red-500" : ""}`}
        >
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
            onClick={() => onDecrement(dimension)}
            disabled={isStreaming}
          >
            <Minus className="h-3.5 w-3.5" />
          </Button>
          <Input
            type="number"
            value={value}
            onChange={e => {
              const v = parseInt(e.target.value);
              if (!isNaN(v)) {
                onChange(dimension, v);
              }
            }}
            disabled={isStreaming}
            className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            min={minValue}
            max={maxValue}
          />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
            onClick={() => onIncrement(dimension)}
            disabled={isStreaming}
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      {error && (
        <p className="text-xs text-red-500 ml-16">{error}</p>
      )}
    </div>
  );
}
