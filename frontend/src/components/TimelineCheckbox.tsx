import { Card, CardContent } from "./ui/card";

interface TimelineCheckboxProps {
  className?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export function TimelineCheckbox({
  className = "",
  checked,
  onChange,
  disabled = false,
}: TimelineCheckboxProps) {
  return (
    <Card className={`bg-transparent border-none shadow-none ${className}`}>
      <CardContent className="p-0">
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={checked}
            onChange={e => onChange(e.target.checked)}
            disabled={disabled}
          />
          <span className="text-sm text-foreground">Show timeline</span>
        </label>
      </CardContent>
    </Card>
  );
}
