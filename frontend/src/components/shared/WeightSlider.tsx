import { Slider } from "../ui/slider";

interface WeightSliderProps {
  value: number;
  onValueChange: (value: number) => void;
  disabled?: boolean;
}

export function WeightSlider({
  value,
  onValueChange,
  disabled = false,
}: WeightSliderProps) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-12">Weight:</span>
      <Slider
        value={[value]}
        onValueChange={([newValue]) => onValueChange(newValue)}
        min={0}
        max={100}
        step={1}
        disabled={disabled}
        className="flex-1"
      />
      <span className="text-xs text-muted-foreground w-12 text-right">
        {value.toFixed(0)}%
      </span>
    </div>
  );
}
