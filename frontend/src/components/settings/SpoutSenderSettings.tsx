import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { Toggle } from "../ui/toggle";
import { Input } from "../ui/input";
import { PARAMETER_METADATA } from "../../data/parameterMetadata";

interface SpoutSenderSettingsProps {
  spoutSender: { enabled: boolean; name: string } | undefined;
  isStreaming: boolean;
  onChange: (sender: { enabled: boolean; name: string }) => void;
}

export function SpoutSenderSettings({
  spoutSender,
  isStreaming,
  onChange,
}: SpoutSenderSettingsProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={PARAMETER_METADATA.spoutSender.label}
          tooltip={PARAMETER_METADATA.spoutSender.tooltip}
          className="text-sm font-medium"
        />
        <Toggle
          pressed={spoutSender?.enabled ?? false}
          onPressedChange={enabled =>
            onChange({
              enabled,
              name: spoutSender?.name ?? "ScopeOut",
            })
          }
          variant="outline"
          size="sm"
          className="h-7"
        >
          {spoutSender?.enabled ? "ON" : "OFF"}
        </Toggle>
      </div>

      {spoutSender?.enabled && (
        <div className="flex items-center gap-3">
          <LabelWithTooltip
            label="Sender Name"
            tooltip="The name of the sender that will send video to Spout-compatible apps like TouchDesigner, Resolume, OBS."
            className="text-xs text-muted-foreground whitespace-nowrap"
          />
          <Input
            type="text"
            value={spoutSender?.name ?? "ScopeOut"}
            onChange={e =>
              onChange({
                enabled: spoutSender?.enabled ?? false,
                name: e.target.value,
              })
            }
            disabled={isStreaming}
            className="h-8 text-sm flex-1"
            placeholder="ScopeOut"
          />
        </div>
      )}
    </div>
  );
}
