/**
 * Dynamic select control for enum parameters.
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import type { SelectControlProps } from "./types";

export function SelectControl({
  value,
  onChange,
  options,
  label,
  tooltip,
  disabled = false,
}: SelectControlProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <LabelWithTooltip
        label={label}
        tooltip={tooltip}
        className="text-sm text-foreground"
      />
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger className="w-[140px] h-7">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(option => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
