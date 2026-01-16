/**
 * Dynamic text input control for string parameters.
 */

import { Input } from "../ui/input";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import type { TextControlProps } from "./types";

export function TextControl({
  value,
  onChange,
  label,
  tooltip,
  disabled = false,
}: TextControlProps) {
  return (
    <div className="flex items-center gap-2">
      <LabelWithTooltip
        label={label}
        tooltip={tooltip}
        className="text-sm text-foreground w-24"
      />
      <Input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
        className="h-8 text-sm flex-1"
      />
    </div>
  );
}
