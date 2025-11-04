import { Slider } from "../ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

interface TemporalTransitionControlsProps {
  transitionSteps: number;
  onTransitionStepsChange: (steps: number) => void;
  temporalInterpolationMethod: "linear" | "slerp";
  onTemporalInterpolationMethodChange: (method: "linear" | "slerp") => void;
  disabled?: boolean;
  showHeader?: boolean;
  showDisabledMessage?: boolean;
  disabledMessage?: string;
  maxSteps?: number;
  className?: string;
}

export function TemporalTransitionControls({
  transitionSteps,
  onTransitionStepsChange,
  temporalInterpolationMethod,
  onTemporalInterpolationMethodChange,
  disabled = false,
  showHeader = false,
  showDisabledMessage = false,
  disabledMessage = "First block cannot have transitions",
  maxSteps = 16,
  className = "",
}: TemporalTransitionControlsProps) {
  return (
    <div className={className}>
      {showHeader && (
        <div className="text-xs font-medium text-muted-foreground mb-3">
          Temporal Transition Settings
        </div>
      )}

      <div
        className={`flex items-center justify-between gap-2 ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <span className="text-xs text-muted-foreground">Temporal Blend:</span>
        <Select
          value={temporalInterpolationMethod}
          onValueChange={value =>
            onTemporalInterpolationMethodChange(value as "linear" | "slerp")
          }
          disabled={disabled}
        >
          <SelectTrigger className="w-24 h-6 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="linear">Linear</SelectItem>
            <SelectItem value="slerp">Slerp</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div
        className={`flex items-center justify-between gap-2 ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <span className="text-xs text-muted-foreground">Transition Steps:</span>
        <div className="flex items-center gap-2 w-32 h-6">
          <Slider
            value={[transitionSteps]}
            onValueChange={([value]) => onTransitionStepsChange(value)}
            min={0}
            max={maxSteps}
            step={1}
            disabled={disabled}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground w-6 text-right">
            {transitionSteps}
          </span>
        </div>
      </div>

      {showDisabledMessage && disabled && (
        <div className="text-xs text-muted-foreground italic mt-2">
          {disabledMessage}
        </div>
      )}
    </div>
  );
}
