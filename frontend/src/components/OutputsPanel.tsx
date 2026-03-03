import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { OutputSinkToggle } from "./OutputSinkToggle";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { SettingsState } from "../types";
import { TempoSyncSection } from "./settings/TempoSyncSection";
import type { TempoState } from "../hooks/useTempoSync";
import type { TempoSourcesResponse, TempoEnableRequest } from "../lib/api";

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
  tempoState?: TempoState;
  tempoSources?: TempoSourcesResponse | null;
  tempoLoading?: boolean;
  tempoError?: string | null;
  onTempoEnable?: (request: TempoEnableRequest) => void;
  onTempoDisable?: () => void;
  onTempoSetBpm?: (bpm: number) => void;
  onTempoRefreshSources?: () => void;
}

export function OutputsPanel({
  className = "",
  outputSinks,
  onOutputSinkChange,
  spoutAvailable = false,
  ndiAvailable = false,
  syphonAvailable = false,
  isStreaming = false,
  tempoState,
  tempoSources,
  tempoLoading = false,
  tempoError = null,
  onTempoEnable,
  onTempoDisable,
  onTempoSetBpm,
  onTempoRefreshSources,
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
      </CardContent>

      {/* Tempo Sync */}
      {tempoState && onTempoEnable && onTempoDisable && (
        <TempoSyncSection
          tempoState={tempoState}
          sources={tempoSources ?? null}
          loading={tempoLoading}
          error={tempoError}
          onEnable={onTempoEnable}
          onDisable={onTempoDisable}
          onSetBpm={onTempoSetBpm}
          onRefreshSources={onTempoRefreshSources ?? (() => {})}
        />
      )}
    </Card>
  );
}
