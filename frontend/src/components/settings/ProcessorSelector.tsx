import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";
import type { PipelineInfo, InputMode } from "../../types";

interface ProcessorSelectorProps {
  type: "preprocessor" | "postprocessor";
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  pipelines: Record<string, PipelineInfo> | null;
  inputMode?: InputMode;
  disabled?: boolean;
}

export function ProcessorSelector({
  type,
  selectedIds,
  onChange,
  pipelines,
  inputMode,
  disabled,
}: ProcessorSelectorProps) {
  const metadata = PARAMETER_METADATA[type];

  const filteredPipelines = Object.entries(pipelines || {}).filter(
    ([, info]) => {
      const matchesUsage = info.usage?.includes(type) ?? false;
      if (!matchesUsage) return false;
      if (type === "preprocessor" && inputMode) {
        return info.supportedModes?.includes(inputMode) ?? false;
      }
      if (type === "postprocessor") {
        return info.supportedModes?.includes("video") ?? false;
      }
      return true;
    }
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={metadata.label}
          tooltip={metadata.tooltip}
          className="text-sm font-medium"
        />
        <Select
          value={selectedIds.length > 0 ? selectedIds[0] : "none"}
          onValueChange={value => onChange(value === "none" ? [] : [value])}
          disabled={disabled}
        >
          <SelectTrigger className="w-[140px] h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            {filteredPipelines.map(([pid]) => (
              <SelectItem key={pid} value={pid}>
                {pid}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
