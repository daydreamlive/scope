/**
 * Dynamic number input control for unbounded or partially bounded numeric parameters.
 * Displays increment/decrement buttons with optional min/max constraints.
 */

import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Plus, Minus } from "lucide-react";
import type { NumberControlProps } from "./types";

export function NumberControl({
  schema,
  value,
  onChange,
  label,
  tooltip,
  disabled = false,
}: NumberControlProps) {
  const [error, setError] = useState<string | null>(null);

  const min = schema.minimum;
  const max = schema.maximum;
  const isInteger = schema.type === "integer";
  const step = isInteger ? 1 : 0.1;

  const validate = (newValue: number): string | null => {
    if (min !== undefined && newValue < min) {
      return `Must be at least ${min}`;
    }
    if (max !== undefined && newValue > max) {
      return `Must be at most ${max}`;
    }
    return null;
  };

  const handleChange = (newValue: number) => {
    const validationError = validate(newValue);
    setError(validationError);
    onChange(newValue);
  };

  const handleIncrement = () => {
    let newValue = value + step;
    if (max !== undefined) {
      newValue = Math.min(max, newValue);
    }
    handleChange(newValue);
  };

  const handleDecrement = () => {
    let newValue = value - step;
    if (min !== undefined) {
      newValue = Math.max(min, newValue);
    }
    handleChange(newValue);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = isInteger
      ? parseInt(e.target.value, 10)
      : parseFloat(e.target.value);
    if (!isNaN(parsed)) {
      handleChange(parsed);
    }
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <LabelWithTooltip
          label={label}
          tooltip={tooltip}
          className="text-sm text-foreground w-24"
        />
        <div
          className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${error ? "border-red-500" : ""}`}
        >
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
            onClick={handleDecrement}
            disabled={disabled}
          >
            <Minus className="h-3.5 w-3.5" />
          </Button>
          <Input
            type="number"
            value={value}
            onChange={handleInputChange}
            disabled={disabled}
            className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            min={min}
            max={max}
            step={step}
          />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
            onClick={handleIncrement}
            disabled={disabled}
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      {error && <p className="text-xs text-red-500 ml-26">{error}</p>}
    </div>
  );
}
