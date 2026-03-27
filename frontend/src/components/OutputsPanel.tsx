import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { OutputSinkToggle } from "./OutputSinkToggle";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { SettingsState } from "../types";
import { trackEvent } from "../lib/analytics";

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
  isStreaming?: boolean;
}

export function OutputsPanel({
  className = "",
  outputSinks,
  onOutputSinkChange,
  spoutAvailable = false,
  ndiAvailable = false,
  syphonAvailable = false,
  isStreaming = false,
}: OutputsPanelProps) {
  const handleOutputSinkChange = (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => {
    onOutputSinkChange?.(sinkType, config);
    trackEvent("output_configured", {
      output_type: sinkType,
      enabled: config.enabled,
      surface: "performance_mode",
    });
  };

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
            onOutputSinkChange={handleOutputSinkChange}
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
            onOutputSinkChange={handleOutputSinkChange}
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
            onOutputSinkChange={handleOutputSinkChange}
            isStreaming={isStreaming}
          />
        )}
      </CardContent>
    </Card>
  );
}
