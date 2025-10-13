import { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { Plus, Minus } from "lucide-react";

interface DenoisingStepsSliderProps {
  className?: string;
  value: number[];
  onChange: (value: number[]) => void;
  disabled?: boolean;
  defaultValues?: number[];
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
}: DenoisingStepsSliderProps) {
  const [localValue, setLocalValue] = useState<number[]>(
    value.length > 0 ? value : defaultValues
  );
  const [validationError, setValidationError] = useState<string>("");
  const updateTimeoutRef = useRef<number | null>(null);

  // Sync with external value changes
  useEffect(() => {
    if (value.length > 0) {
      setLocalValue(value);
    }
  }, [value]);

  // Debounced update function
  const debouncedOnChange = (newValue: number[]) => {
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }

    updateTimeoutRef.current = setTimeout(() => {
      onChange(newValue);
    }, 150); // 150ms delay
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

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

  const handleSliderChange = (index: number, newValue: number[]) => {
    const updatedValue = [...localValue];
    const attemptedValue = newValue[0];
    updatedValue[index] = attemptedValue;

    const error = validateSteps(updatedValue);
    setValidationError(error);

    if (!error) {
      setLocalValue(updatedValue);
      debouncedOnChange(updatedValue);
    } else {
      const boundaryValue = calculateBoundaryValue(index, attemptedValue);
      const clampedValue = Math.max(
        MIN_VALUE,
        Math.min(MAX_VALUE, boundaryValue)
      );
      const boundedValue = [...localValue];
      boundedValue[index] = clampedValue;
      setLocalValue(boundedValue);
      debouncedOnChange(boundedValue);
    }
  };

  const handleValueChange = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    const clampedValue = Math.max(MIN_VALUE, Math.min(MAX_VALUE, newValue));
    updatedValue[index] = clampedValue;

    const error = validateSteps(updatedValue);
    setValidationError(error);

    if (!error) {
      setLocalValue(updatedValue);
      onChange(updatedValue); // Immediate update for discrete actions
    } else {
      const boundaryValue = calculateBoundaryValue(index, clampedValue);
      const finalValue = Math.max(
        MIN_VALUE,
        Math.min(MAX_VALUE, boundaryValue)
      );
      const boundedValue = [...localValue];
      boundedValue[index] = finalValue;
      setLocalValue(boundedValue);
      onChange(boundedValue);
    }
  };

  const incrementValue = (index: number) => {
    const newValue = Math.min(MAX_VALUE, localValue[index] + 1);
    handleValueChange(index, newValue);
  };

  const decrementValue = (index: number) => {
    const newValue = Math.max(MIN_VALUE, localValue[index] - 1);
    handleValueChange(index, newValue);
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
        <label className="text-sm text-foreground">Denoising Step List</label>
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
          <div key={index} className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm text-foreground w-16">
                Step {index + 1}:
              </label>
              <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8 min-w-0">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                  onClick={() => decrementValue(index)}
                  disabled={disabled}
                >
                  <Minus className="h-3.5 w-3.5" />
                </Button>
                <Input
                  type="number"
                  value={stepValue}
                  onChange={e =>
                    handleValueChange(
                      index,
                      parseInt(e.target.value) || MIN_VALUE
                    )
                  }
                  disabled={disabled}
                  className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 min-w-0 px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                  min={MIN_VALUE}
                  max={MAX_VALUE}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                  onClick={() => incrementValue(index)}
                  disabled={disabled}
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
                {localValue.length > MIN_SLIDERS && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-destructive/10 text-destructive"
                    onClick={() => removeSlider(index)}
                    disabled={disabled}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                )}
              </div>
            </div>
            <Slider
              value={[stepValue]}
              onValueChange={value => handleSliderChange(index, value)}
              min={MIN_VALUE}
              max={MAX_VALUE}
              step={1}
              disabled={disabled}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
