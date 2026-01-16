/**
 * Dynamic toggle control for boolean parameters.
 */

import { Toggle } from "../ui/toggle";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import type { ToggleControlProps } from "./types";

export function ToggleControl({
  value,
  onChange,
  label,
  tooltip,
  disabled = false,
}: ToggleControlProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <LabelWithTooltip
        label={label}
        tooltip={tooltip}
        className="text-sm text-foreground"
      />
      <Toggle
        pressed={value}
        onPressedChange={onChange}
        variant="outline"
        size="sm"
        className="h-7"
        disabled={disabled}
      >
        {value ? "ON" : "OFF"}
      </Toggle>
    </div>
  );
}
