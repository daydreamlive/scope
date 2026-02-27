import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { SliderWithInput } from "./ui/slider-with-input";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Plus, Trash2 } from "lucide-react";
import { MIDIMappable } from "./MIDIMappable";

interface DenoisingStepsSliderProps {
  className?: string;
  value: number[];
  onChange: (value: number[]) => void;
  disabled?: boolean;
  defaultValues?: number[];
  tooltip?: string;
}

const MIN_SLIDERS = 1;
const MAX_SLIDERS = 10;
const MIN_VALUE = 0;
const MAX_VALUE = 1000;
const DEFAULT_VALUES = [700, 500];

export function DenoisingStepsSlider({
  className = "",
  value,
  onChange,
  disabled = false,
  defaultValues = DEFAULT_VALUES,
  tooltip,
}: DenoisingStepsSliderProps) {
  const [localValue, setLocalValue] = useState<number[]>(
    value.length > 0 ? value : defaultValues
  );
  const [validationError, setValidationError] = useState<string>("");

  // Sync with external value changes
  useEffect(() => {
    if (value.length > 0) {
      setLocalValue(value);
    }
  }, [value]);

  const validateSteps = (steps: number[]): string => {
    for (let i = 1; i < steps.length; i++) {
      if (steps[i] >= steps[i - 1]) {
        return `Step ${i + 1} must be lower than Step ${i}`;
      }
    }
    return "";
  };

  const handleStepValueChange = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    updatedValue[index] = newValue;

    // Ensure descending order constraint by pushing sliders in both directions
    // First, push lower sliders down if they violate the constraint
    for (let i = index + 1; i < updatedValue.length; i++) {
      if (updatedValue[i] >= updatedValue[i - 1]) {
        updatedValue[i] = Math.max(MIN_VALUE, updatedValue[i - 1] - 1);
      }
    }
    // Then, push higher sliders up if they violate the constraint
    for (let i = index - 1; i >= 0; i--) {
      if (updatedValue[i] <= updatedValue[i + 1]) {
        updatedValue[i] = Math.min(MAX_VALUE, updatedValue[i + 1] + 1);
      }
    }

    // Clamp the updated value to valid range
    updatedValue[index] = Math.max(
      MIN_VALUE,
      Math.min(MAX_VALUE, updatedValue[index])
    );

    const error = validateSteps(updatedValue);
    setValidationError(error);
    setLocalValue(updatedValue);
  };

  const handleStepCommit = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    updatedValue[index] = newValue;
    onChange(updatedValue);
  };

  const addSlider = () => {
    if (localValue.length < MAX_SLIDERS) {
      // Add a new slider with a value lower than the last one
      const lastValue = localValue[localValue.length - 1];
      const newValue = Math.max(MIN_VALUE, lastValue - 100);
      const updatedValue = [...localValue, newValue];

      const error = validateSteps(updatedValue);
      setValidationError(error);

      if (!error) {
        setLocalValue(updatedValue);
        onChange(updatedValue);
      }
    }
  };

  const removeSlider = (index: number) => {
    if (localValue.length > MIN_SLIDERS) {
      const updatedValue = localValue.filter((_, i) => i !== index);

      const error = validateSteps(updatedValue);
      setValidationError(error);

      if (!error) {
        setLocalValue(updatedValue);
        onChange(updatedValue);
      }
    }
  };

  const resetToDefaults = () => {
    setValidationError("");
    setLocalValue(defaultValues);
    onChange(defaultValues);
  };

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between">
        <LabelWithTooltip
          label="Denoising Step List"
          tooltip={tooltip}
          className="text-sm font-medium"
        />
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={resetToDefaults}
            disabled={disabled}
            className="h-7 px-2 text-xs"
          >
            Reset
          </Button>
          <MIDIMappable actionId="add_denoising_step">
            <Button
              variant="outline"
              size="sm"
              onClick={addSlider}
              disabled={disabled || localValue.length >= MAX_SLIDERS}
              className="h-7 w-7 p-0"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </MIDIMappable>
        </div>
      </div>

      {validationError && (
        <div className="flex items-center gap-2 p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
          <span>{validationError}</span>
        </div>
      )}

      <div className="space-y-3">
        {localValue.map((stepValue, index) => (
          <MIDIMappable
            key={index}
            parameterId="denoising_step_list"
            arrayIndex={index}
          >
            <SliderWithInput
              label={`Step ${index + 1}`}
              labelClassName="text-xs text-muted-foreground w-12"
              value={stepValue}
              onValueChange={value => handleStepValueChange(index, value)}
              onValueCommit={value => handleStepCommit(index, value)}
              min={MIN_VALUE}
              max={MAX_VALUE}
              step={1}
              incrementAmount={1}
              disabled={disabled}
              inputParser={v => parseInt(v) || MIN_VALUE}
              renderExtraButton={() =>
                localValue.length > MIN_SLIDERS ? (
                  <MIDIMappable actionId="remove_denoising_step">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0 rounded-none hover:bg-destructive/10 text-destructive"
                      onClick={() => removeSlider(index)}
                      disabled={disabled}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </MIDIMappable>
                ) : null
              }
            />
          </MIDIMappable>
        ))}
      </div>
    </div>
  );
}
