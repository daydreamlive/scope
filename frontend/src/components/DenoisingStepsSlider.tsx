import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { SliderWithInput } from "./ui/slider-with-input";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Plus, Trash2 } from "lucide-react";

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

  const calculateBoundaryValue = (
    index: number,
    attemptedValue: number
  ): number => {
    // If we violated the constraint with the previous step, set to previous step - 1
    if (index > 0 && attemptedValue >= localValue[index - 1]) {
      return localValue[index - 1] - 1;
    }
    // If we violated the constraint with the next step, set to next step + 1
    if (
      index < localValue.length - 1 &&
      attemptedValue <= localValue[index + 1]
    ) {
      return localValue[index + 1] + 1;
    }
    return attemptedValue;
  };

  const handleStepValueChange = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    updatedValue[index] = newValue;

    const error = validateSteps(updatedValue);
    setValidationError(error);

    if (!error) {
      setLocalValue(updatedValue);
    } else {
      const boundaryValue = calculateBoundaryValue(index, newValue);
      const clampedValue = Math.max(
        MIN_VALUE,
        Math.min(MAX_VALUE, boundaryValue)
      );
      const boundedValue = [...localValue];
      boundedValue[index] = clampedValue;
      setLocalValue(boundedValue);
    }
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
          className="text-sm text-foreground"
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
          <Button
            variant="outline"
            size="sm"
            onClick={addSlider}
            disabled={disabled || localValue.length >= MAX_SLIDERS}
            className="h-7 w-7 p-0"
          >
            <Plus className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {validationError && (
        <div className="flex items-center gap-2 p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
          <span>{validationError}</span>
        </div>
      )}

      <div className="space-y-3">
        {localValue.map((stepValue, index) => (
          <SliderWithInput
            key={index}
            label={`Step ${index + 1}:`}
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
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-none hover:bg-destructive/10 text-destructive"
                  onClick={() => removeSlider(index)}
                  disabled={disabled}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              ) : null
            }
          />
        ))}
      </div>
    </div>
  );
}
