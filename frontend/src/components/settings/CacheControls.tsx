import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Toggle } from "../ui/toggle";
import { Button } from "../ui/button";
import { SliderWithInput } from "../ui/slider-with-input";
import { RotateCcw } from "lucide-react";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";
import type { useLocalSliderValue } from "../../hooks/useLocalSliderValue";

interface CacheControlsProps {
  manageCache: boolean;
  onManageCacheChange: (enabled: boolean) => void;
  onResetCache: () => void;
  kvCacheAttentionBiasSlider: ReturnType<typeof useLocalSliderValue>;
  supportsKvCacheBias?: boolean;
}

export function CacheControls({
  manageCache,
  onManageCacheChange,
  onResetCache,
  kvCacheAttentionBiasSlider,
  supportsKvCacheBias,
}: CacheControlsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="space-y-2 pt-2">
          {supportsKvCacheBias && (
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
              labelClassName="text-sm font-medium w-20"
              valueFormatter={kvCacheAttentionBiasSlider.formatValue}
              inputParser={v => parseFloat(v) || 1.0}
            />
          )}

          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={PARAMETER_METADATA.manageCache.label}
              tooltip={PARAMETER_METADATA.manageCache.tooltip}
              className="text-sm font-medium"
            />
            <Toggle
              pressed={manageCache}
              onPressedChange={onManageCacheChange}
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
              className="text-sm font-medium"
            />
            <Button
              type="button"
              onClick={onResetCache}
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
  );
}
