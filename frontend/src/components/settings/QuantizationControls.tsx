import { LabelWithTooltip } from "../ui/label-with-tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";

interface QuantizationControlsProps {
  quantization: "fp8_e4m3fn" | null;
  onQuantizationChange: (q: "fp8_e4m3fn" | null) => void;
  vaceEnabled: boolean;
  isStreaming: boolean;
}

export function QuantizationControls({
  quantization,
  onQuantizationChange,
  vaceEnabled,
  isStreaming,
}: QuantizationControlsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={PARAMETER_METADATA.quantization.label}
              tooltip={PARAMETER_METADATA.quantization.tooltip}
              className="text-sm font-medium"
            />
            <Select
              value={quantization || "none"}
              onValueChange={value =>
                onQuantizationChange(
                  value === "none" ? null : (value as "fp8_e4m3fn")
                )
              }
              disabled={isStreaming || vaceEnabled}
            >
              <SelectTrigger className="w-[140px] h-7">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="fp8_e4m3fn">fp8_e4m3fn (Dynamic)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {vaceEnabled && (
            <p className="text-xs text-muted-foreground">
              Disabled because VACE is enabled. Disable VACE to use FP8
              quantization.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
