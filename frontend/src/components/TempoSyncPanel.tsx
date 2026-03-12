import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { TempoSyncSection } from "./settings/TempoSyncSection";
import type { ModulationsState } from "./settings/ModulationSection";
import type { TempoState } from "../hooks/useTempoSync";
import type {
  TempoSourcesResponse,
  TempoEnableRequest,
  PipelineConfigSchema,
} from "../lib/api";

interface TempoSyncPanelProps {
  className?: string;
  tempoState: TempoState;
  tempoSources?: TempoSourcesResponse | null;
  tempoLoading?: boolean;
  tempoError?: string | null;
  onTempoEnable: (request: TempoEnableRequest) => void;
  onTempoDisable: () => void;
  onTempoSetBpm?: (bpm: number) => void;
  onTempoRefreshSources?: () => void;
  quantizeMode?: string;
  onQuantizeModeChange?: (mode: string) => void;
  lookaheadMs?: number;
  onLookaheadMsChange?: (ms: number) => void;
  modulations?: ModulationsState;
  onModulationsChange?: (modulations: ModulationsState) => void;
  configSchema?: PipelineConfigSchema;
  beatCacheResetRate?: string;
  onBeatCacheResetRateChange?: (rate: string) => void;
  promptCycleRate?: string;
  onPromptCycleRateChange?: (rate: string) => void;
}

export function TempoSyncPanel({
  className = "",
  tempoState,
  tempoSources,
  tempoLoading = false,
  tempoError = null,
  onTempoEnable,
  onTempoDisable,
  onTempoSetBpm,
  onTempoRefreshSources,
  quantizeMode,
  onQuantizeModeChange,
  lookaheadMs,
  onLookaheadMsChange,
  modulations,
  onModulationsChange,
  configSchema,
  beatCacheResetRate,
  onBeatCacheResetRateChange,
  promptCycleRate,
  onPromptCycleRateChange,
}: TempoSyncPanelProps) {
  return (
    <Card className={className}>
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-base font-medium">Tempo Sync</CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0">
        <TempoSyncSection
          tempoState={tempoState}
          sources={tempoSources ?? null}
          loading={tempoLoading}
          error={tempoError}
          onEnable={onTempoEnable}
          onDisable={onTempoDisable}
          onSetBpm={onTempoSetBpm}
          onRefreshSources={onTempoRefreshSources ?? (() => {})}
          quantizeMode={quantizeMode}
          onQuantizeModeChange={onQuantizeModeChange}
          lookaheadMs={lookaheadMs}
          onLookaheadMsChange={onLookaheadMsChange}
          modulations={modulations}
          onModulationsChange={onModulationsChange}
          configSchema={configSchema}
          beatCacheResetRate={beatCacheResetRate}
          onBeatCacheResetRateChange={onBeatCacheResetRateChange}
          promptCycleRate={promptCycleRate}
          onPromptCycleRateChange={onPromptCycleRateChange}
        />
      </CardContent>
    </Card>
  );
}
