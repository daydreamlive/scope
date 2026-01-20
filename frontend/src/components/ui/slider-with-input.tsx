import { Button } from "./button";
import { Input } from "./input";
import { DebouncedSlider } from "./debounced-slider";
import { LabelWithTooltip } from "./label-with-tooltip";
import { Plus, Minus } from "lucide-react";

interface SliderWithInputProps {
  label?: string;
  tooltip?: string;
  value: number;
  onValueChange: (value: number) => void;
  onValueCommit?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  incrementAmount?: number;
  disabled?: boolean;
  className?: string;
  labelClassName?: string;
  debounceMs?: number;
  valueFormatter?: (value: number) => number;
  inputParser?: (value: string) => number;
  renderExtraButton?: () => React.ReactNode;
}

/**
 * A reusable component that combines a labeled input field with increment/decrement buttons
 * and a debounced slider for numeric value selection.
 *
 * Features:
 * - Input field with increment/decrement buttons for precise control
 * - Debounced slider for smooth dragging without excessive callbacks
 * - Optional value formatting and parsing for custom number handling
 * - Optional extra button rendering (e.g., remove button)
 */
export function SliderWithInput({
  label,
  tooltip,
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 100,
  step = 1,
  incrementAmount = step,
  disabled = false,
  className = "",
  labelClassName = "text-sm text-foreground w-16",
  debounceMs = 100,
  valueFormatter = v => v,
  inputParser = v => {
    const parsed = parseFloat(v);
    return isNaN(parsed) ? min : parsed;
  },
  renderExtraButton,
}: SliderWithInputProps) {
  const handleIncrement = () => {
    const newValue = Math.min(max, value + incrementAmount);
    const formattedValue = valueFormatter(newValue);
    onValueChange(formattedValue);
    onValueCommit?.(formattedValue);
  };

  const handleDecrement = () => {
    const newValue = Math.max(min, value - incrementAmount);
    const formattedValue = valueFormatter(newValue);
    onValueChange(formattedValue);
    onValueCommit?.(formattedValue);
  };

  const handleInputChange = (inputValue: string) => {
    const parsedValue = inputParser(inputValue);
    const clampedValue = Math.max(min, Math.min(max, parsedValue));
    const formattedValue = valueFormatter(clampedValue);
    onValueChange(formattedValue);
    onValueCommit?.(formattedValue);
  };

  const handleSliderValueChange = (newValue: number[]) => {
    const formattedValue = valueFormatter(newValue[0]);
    onValueChange(formattedValue);
  };

  const handleSliderCommit = (newValue: number[]) => {
    const formattedValue = valueFormatter(newValue[0]);
    onValueCommit?.(formattedValue);
  };

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center gap-2">
        {label && (
          <LabelWithTooltip
            label={label}
            tooltip={tooltip}
            className={labelClassName}
          />
        )}
        <div className="flex-1 flex items-center border rounded-full overflow-hidden h-8 min-w-0">
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
            onChange={e => handleInputChange(e.target.value)}
            disabled={disabled}
            className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 min-w-0 px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
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
          {renderExtraButton?.()}
        </div>
      </div>
      <DebouncedSlider
        value={[value]}
        onValueChange={handleSliderValueChange}
        onValueCommit={handleSliderCommit}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full"
        debounceMs={debounceMs}
      />
    </div>
  );
}
