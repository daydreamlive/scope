import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Toggle } from "../ui/toggle";
import { SliderWithInput } from "../ui/slider-with-input";
import { Info } from "lucide-react";
import type { useLocalSliderValue } from "../../hooks/useLocalSliderValue";
import type { InputMode } from "../../types";

interface VACEControlsProps {
  vaceEnabled: boolean;
  onVaceEnabledChange: (enabled: boolean) => void;
  vaceUseInputVideo: boolean;
  onVaceUseInputVideoChange: (enabled: boolean) => void;
  vaceContextScaleSlider: ReturnType<typeof useLocalSliderValue>;
  quantization: string | null;
  inputMode?: InputMode;
  isStreaming: boolean;
  isLoading: boolean;
}

export function VACEControls({
  vaceEnabled,
  onVaceEnabledChange,
  vaceUseInputVideo,
  onVaceUseInputVideoChange,
  vaceContextScaleSlider,
  quantization,
  inputMode,
  isStreaming,
  isLoading,
}: VACEControlsProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label="VACE"
          tooltip="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, you can use reference images for R2V generation. In Video input mode, a separate toggle controls whether the input video is used for VACE conditioning or for latent initialization. Requires pipeline reload to take effect."
          className="text-sm font-medium"
        />
        <Toggle
          pressed={vaceEnabled}
          onPressedChange={onVaceEnabledChange}
          variant="outline"
          size="sm"
          className="h-7"
          disabled={isStreaming || isLoading}
        >
          {vaceEnabled ? "ON" : "OFF"}
        </Toggle>
      </div>
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
              onPressedChange={onVaceUseInputVideoChange}
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
              label="Scale"
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
  );
}
