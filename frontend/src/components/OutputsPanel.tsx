import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Toggle } from "./ui/toggle";
import { OutputSinkToggle } from "./OutputSinkToggle";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { SettingsState } from "../types";

interface OutputsPanelProps {
  className?: string;
  outputSinks?: SettingsState["outputSinks"];
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;
  spoutAvailable?: boolean;
  ndiAvailable?: boolean;
  syphonAvailable?: boolean;
  oscEnabled?: boolean;
  onOscToggle?: (enabled: boolean) => void;
  isStreaming?: boolean;
}

export function OutputsPanel({
  className = "",
  outputSinks,
  onOutputSinkChange,
  spoutAvailable = false,
  ndiAvailable = false,
  syphonAvailable = false,
  oscEnabled = false,
  onOscToggle,
  isStreaming = false,
}: OutputsPanelProps) {
  return (
    <Card className={className}>
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-base font-medium">Outputs</CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0 space-y-3">
        {spoutAvailable && (
          <OutputSinkToggle
            sinkType="spout"
            label="Spout Output"
            tooltip="Send video to Spout-compatible apps (Windows) like TouchDesigner, Resolume, OBS."
            senderNameTooltip="The name visible to Spout receivers."
            defaultName="ScopeOut"
            enabled={outputSinks?.spout?.enabled ?? false}
            name={outputSinks?.spout?.name ?? "ScopeOut"}
            onOutputSinkChange={onOutputSinkChange}
            isStreaming={isStreaming}
          />
        )}

        {ndiAvailable && (
          <OutputSinkToggle
            sinkType="ndi"
            label={PARAMETER_METADATA.ndiSender.label}
            tooltip={PARAMETER_METADATA.ndiSender.tooltip}
            senderNameTooltip="The name visible to NDI receivers on the network."
            defaultName="Scope"
            enabled={outputSinks?.ndi?.enabled ?? false}
            name={outputSinks?.ndi?.name ?? "Scope"}
            onOutputSinkChange={onOutputSinkChange}
            isStreaming={isStreaming}
          />
        )}

        {syphonAvailable && (
          <OutputSinkToggle
            sinkType="syphon"
            label={PARAMETER_METADATA.syphonSender.label}
            tooltip={PARAMETER_METADATA.syphonSender.tooltip}
            senderNameTooltip="The name visible to Syphon receivers on this Mac."
            defaultName="Scope"
            enabled={outputSinks?.syphon?.enabled ?? false}
            name={outputSinks?.syphon?.name ?? "Scope"}
            onOutputSinkChange={onOutputSinkChange}
            isStreaming={isStreaming}
          />
        )}

        <div className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label="OSC Control"
              tooltip="Open Sound Control server for external parameter control from TouchDesigner, Resolume, Max/MSP, etc."
              className="text-sm font-medium"
            />
            <Toggle
              pressed={oscEnabled}
              onPressedChange={onOscToggle ?? (() => {})}
              variant="outline"
              size="sm"
              className="h-7"
            >
              {oscEnabled ? "ON" : "OFF"}
            </Toggle>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
