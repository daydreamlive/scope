import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Toggle } from "../ui/toggle";
import { SliderWithInput } from "../ui/slider-with-input";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";
import type { useLocalSliderValue } from "../../hooks/useLocalSliderValue";

interface NoiseControlsProps {
  noiseController: boolean;
  onNoiseControllerChange: (enabled: boolean) => void;
  noiseScaleSlider: ReturnType<typeof useLocalSliderValue>;
  isStreaming: boolean;
}

export function NoiseControls({
  noiseController,
  onNoiseControllerChange,
  noiseScaleSlider,
  isStreaming,
}: NoiseControlsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={PARAMETER_METADATA.noiseController.label}
              tooltip={PARAMETER_METADATA.noiseController.tooltip}
              className="text-sm font-medium"
            />
            <Toggle
              pressed={noiseController}
              onPressedChange={onNoiseControllerChange}
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
          labelClassName="text-sm font-medium w-20"
          valueFormatter={noiseScaleSlider.formatValue}
          inputParser={v => parseFloat(v) || 0.0}
        />
      </div>
    </div>
  );
}
