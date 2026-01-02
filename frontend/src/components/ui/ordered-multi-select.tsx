import * as React from "react";
import { Check, ChevronDown, ChevronUp, X } from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "./button";

interface OrderedMultiSelectProps {
  options: Array<{ value: string; label: string }>;
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function OrderedMultiSelect({
  options,
  value,
  onChange,
  placeholder = "Select...",
  disabled = false,
  className,
}: OrderedMultiSelectProps) {
  const [open, setOpen] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };

    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [open]);

  const toggleOption = (optionValue: string) => {
    if (disabled) return;
    const newValue = value.includes(optionValue)
      ? value.filter((v) => v !== optionValue)
      : [...value, optionValue];
    onChange(newValue);
  };

  const removeItem = (index: number) => {
    if (disabled) return;
    const newValue = value.filter((_, i) => i !== index);
    onChange(newValue);
  };

  const moveUp = (index: number) => {
    if (disabled || index === 0) return;
    const newValue = [...value];
    [newValue[index - 1], newValue[index]] = [newValue[index], newValue[index - 1]];
    onChange(newValue);
  };

  const moveDown = (index: number) => {
    if (disabled || index === value.length - 1) return;
    const newValue = [...value];
    [newValue[index], newValue[index + 1]] = [newValue[index + 1], newValue[index]];
    onChange(newValue);
  };

  const getLabel = (value: string) => {
    return options.find((opt) => opt.value === value)?.label || value;
  };

  const availableOptions = options.filter((opt) => !value.includes(opt.value));

  return (
    <div ref={containerRef} className={cn("space-y-2", className)}>
      {/* Selected items in order */}
      {value.length > 0 && (
        <div className="space-y-1">
          {value.map((itemValue, index) => (
            <div
              key={`${itemValue}-${index}`}
              className="flex items-center gap-1 rounded-md border bg-background px-2 py-1.5 text-sm"
            >
              <div className="flex items-center gap-1 flex-1 min-w-0">
                <span className="text-muted-foreground text-xs w-4 shrink-0">
                  {index + 1}.
                </span>
                <span className="truncate">{getLabel(itemValue)}</span>
              </div>
              <div className="flex items-center gap-0.5 shrink-0">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => moveUp(index)}
                  disabled={disabled || index === 0}
                  title="Move up"
                >
                  <ChevronUp className="h-3 w-3" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => moveDown(index)}
                  disabled={disabled || index === value.length - 1}
                  title="Move down"
                >
                  <ChevronDown className="h-3 w-3" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => removeItem(index)}
                  disabled={disabled}
                  title="Remove"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Dropdown to add more */}
      <div className="relative">
        <Button
          type="button"
          variant="outline"
          className={cn(
            "flex h-7 w-full items-center justify-between px-3 text-sm",
            value.length === 0 && "text-muted-foreground"
          )}
          onClick={() => !disabled && setOpen(!open)}
          disabled={disabled || availableOptions.length === 0}
        >
          <span className="truncate">
            {availableOptions.length === 0
              ? "All preprocessors selected"
              : placeholder}
          </span>
          <ChevronDown
            className={cn(
              "ml-2 h-4 w-4 shrink-0 opacity-50 transition-transform",
              open && "rotate-180"
            )}
          />
        </Button>
        {open && availableOptions.length > 0 && (
          <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover shadow-md">
            <div className="max-h-60 overflow-auto p-1">
              {availableOptions.map((option) => (
                <div
                  key={option.value}
                  className={cn(
                    "relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground"
                  )}
                  onClick={() => {
                    toggleOption(option.value);
                    setOpen(false);
                  }}
                >
                  <div className="flex h-4 w-4 items-center justify-center mr-2">
                    <Check className="h-4 w-4 opacity-0" />
                  </div>
                  <span>{option.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
