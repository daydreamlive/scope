import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { MIDIMappable } from "./MIDIMappable";
import { cn } from "../lib/utils";
import { Check } from "lucide-react";

interface PromptSlot {
  text: string;
}

interface PromptSlotsProps {
  className?: string;
  slots: PromptSlot[];
  activeSlotIndex: number;
  onSlotsChange: (slots: PromptSlot[]) => void;
  onActiveSlotChange: (index: number) => void;
  disabled?: boolean;
}

export function PromptSlots({
  className = "",
  slots,
  activeSlotIndex,
  onSlotsChange,
  onActiveSlotChange,
  disabled = false,
}: PromptSlotsProps) {
  const handleSlotTextChange = (index: number, text: string) => {
    const updatedSlots = [...slots];
    updatedSlots[index] = { ...updatedSlots[index], text };
    onSlotsChange(updatedSlots);
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">Prompt Slots</h3>
      </div>

      <div className="space-y-2">
        {slots.map((slot, index) => (
          <div
            key={index}
            className={cn(
              "flex items-center gap-2 p-2 rounded-md border transition-colors",
              activeSlotIndex === index
                ? "border-primary bg-primary/5"
                : "border-border bg-card"
            )}
          >
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-6 w-6 shrink-0 rounded-full p-0",
                activeSlotIndex === index
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              )}
              onClick={() => !disabled && onActiveSlotChange(index)}
              disabled={disabled}
              title={`Select prompt ${index + 1}`}
            >
              {activeSlotIndex === index && (
                <Check className="h-3.5 w-3.5" />
              )}
            </Button>
            <MIDIMappable
              actionId={`switch_prompt_${index}`}
              className="flex-1"
            >
              <Input
                value={slot.text}
                onChange={(e) => handleSlotTextChange(index, e.target.value)}
                placeholder={`Prompt ${index + 1}`}
                disabled={disabled}
                className="w-full"
              />
            </MIDIMappable>
          </div>
        ))}
      </div>
    </div>
  );
}
