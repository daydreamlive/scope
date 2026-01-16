/**
 * Dynamic slider control for bounded numeric parameters.
 * Infers min/max/step from JSON schema constraints.
 */

import { SliderWithInput } from "../ui/slider-with-input";
import { useLocalSliderValue } from "../../hooks/useLocalSliderValue";
import type { SliderControlProps } from "./types";

/**
 * Determines an appropriate step value based on the range and type.
 */
function inferStep(min: number, max: number, isInteger: boolean): number {
  if (isInteger) {
    return 1;
  }

  const range = max - min;
  if (range <= 1) return 0.01;
  if (range <= 10) return 0.1;
  if (range <= 100) return 1;
  return Math.floor(range / 100);
}

export function SliderControl({
  schema,
  value,
  onChange,
  label,
  tooltip,
  disabled = false,
}: SliderControlProps) {
  const min = schema.minimum ?? 0;
  const max = schema.maximum ?? 100;
  const isInteger = schema.type === "integer";
  const step = inferStep(min, max, isInteger);

  // Use local slider value hook for smooth dragging
  const slider = useLocalSliderValue(value, onChange);

  // Value formatter for display
  const valueFormatter = (v: number) => {
    if (isInteger) {
      return Math.round(v);
    }
    // Round to step precision
    const precision = step < 1 ? Math.ceil(-Math.log10(step)) : 0;
    return Number(v.toFixed(precision));
  };

  // Input parser
  const inputParser = (v: string) => {
    const parsed = isInteger ? parseInt(v, 10) : parseFloat(v);
    return isNaN(parsed) ? min : parsed;
  };

  return (
    <SliderWithInput
      label={label}
      tooltip={tooltip}
      value={slider.localValue}
      onValueChange={slider.handleValueChange}
      onValueCommit={slider.handleValueCommit}
      min={min}
      max={max}
      step={step}
      incrementAmount={step}
      disabled={disabled}
      labelClassName="text-sm text-foreground w-24"
      valueFormatter={valueFormatter}
      inputParser={inputParser}
    />
  );
}
