import { Input } from "./ui/input";
import { Toggle } from "./ui/toggle";
import { LabelWithTooltip } from "./ui/label-with-tooltip";

export type OutputSinkType = "spout" | "ndi" | "syphon";

interface OutputSinkToggleProps {
  sinkType: OutputSinkType;
  label: string;
  tooltip: string;
  senderNameTooltip: string;
  defaultName: string;
  enabled: boolean;
  name: string;
  onOutputSinkChange?: (
    sinkType: OutputSinkType,
    config: { enabled: boolean; name: string }
  ) => void;
  isStreaming?: boolean;
}

export function OutputSinkToggle({
  sinkType,
  label,
  tooltip,
  senderNameTooltip,
  defaultName,
  enabled,
  name,
  onOutputSinkChange,
  isStreaming = false,
}: OutputSinkToggleProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={label}
          tooltip={tooltip}
          className="text-sm font-medium"
        />
        <Toggle
          pressed={enabled}
          onPressedChange={newEnabled => {
            onOutputSinkChange?.(sinkType, {
              enabled: newEnabled,
              name,
            });
          }}
          variant="outline"
          size="sm"
          className="h-7"
        >
          {enabled ? "ON" : "OFF"}
        </Toggle>
      </div>

      {enabled && (
        <div className="flex items-center gap-3">
          <LabelWithTooltip
            label="Sender Name"
            tooltip={senderNameTooltip}
            className="text-xs text-muted-foreground whitespace-nowrap"
          />
          <Input
            type="text"
            value={name}
            onChange={e => {
              onOutputSinkChange?.(sinkType, {
                enabled,
                name: e.target.value,
              });
            }}
            disabled={isStreaming}
            className="h-8 text-sm flex-1"
            placeholder={defaultName}
          />
        </div>
      )}
    </div>
  );
}
