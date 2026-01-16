import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Badge } from "./ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { Hammer, Info } from "lucide-react";
import type {
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  InputMode,
  PipelineInfo,
  VaeType,
} from "../types";
import { SettingsPanelRenderer } from "./SettingsPanelRenderer";
import type { PipelineSchemaInfo } from "../lib/api";

interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  // Pipeline schema info - required for SettingsPanelRenderer
  schema?: PipelineSchemaInfo;
  // Resolution is required - parent should always provide from schema defaults
  resolution: {
    height: number;
    width: number;
  };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (denoisingSteps: number[]) => void;
  // Default denoising steps for reset functionality - derived from backend schema
  defaultDenoisingSteps: number[];
  noiseScale?: number;
  onNoiseScaleChange?: (noiseScale: number) => void;
  noiseController?: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  manageCache?: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  quantization?: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
  kvCacheAttentionBias?: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;
  onResetCache?: () => void;
  loras?: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  loraMergeStrategy?: LoraMergeStrategy;
  // Input mode for conditional rendering of noise controls
  inputMode?: InputMode;
  // Spout settings
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
  // Whether Spout is available (server-side detection for native Windows, not WSL)
  spoutAvailable?: boolean;
  // VACE settings
  vaceEnabled?: boolean;
  onVaceEnabledChange?: (enabled: boolean) => void;
  vaceUseInputVideo?: boolean;
  onVaceUseInputVideoChange?: (enabled: boolean) => void;
  vaceContextScale?: number;
  onVaceContextScaleChange?: (scale: number) => void;
  // VAE type selection
  vaeType?: VaeType;
  onVaeTypeChange?: (vaeType: VaeType) => void;
  // Available VAE types from backend registry
  vaeTypes?: string[];
  // Preprocessors
  preprocessorIds?: string[];
  onPreprocessorIdsChange?: (ids: string[]) => void;
}

export function SettingsPanel({
  className = "",
  pipelines,
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  isLoading = false,
  schema,
  resolution,
  onResolutionChange,
  seed = 42,
  onSeedChange,
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
  defaultDenoisingSteps,
  noiseScale = 0.7,
  onNoiseScaleChange,
  noiseController = true,
  onNoiseControllerChange,
  manageCache = true,
  onManageCacheChange,
  quantization = "fp8_e4m3fn",
  onQuantizationChange,
  kvCacheAttentionBias = 0.3,
  onKvCacheAttentionBiasChange,
  onResetCache,
  loras = [],
  onLorasChange,
  loraMergeStrategy = "permanent_merge",
  inputMode,
  spoutSender,
  onSpoutSenderChange,
  spoutAvailable = false,
  vaceEnabled = true,
  onVaceEnabledChange,
  vaceUseInputVideo = true,
  onVaceUseInputVideoChange,
  vaceContextScale = 1.0,
  onVaceContextScaleChange,
  vaeType = "wan",
  onVaeTypeChange,
  vaeTypes,
  preprocessorIds = [],
  onPreprocessorIdsChange,
}: SettingsPanelProps) {
  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const currentPipeline = pipelines?.[pipelineId];

  // Get settings panel configuration from schema
  // Use mode-specific settings_panel if available, otherwise use base settings_panel
  const settingsPanelConfig =
    schema?.mode_defaults?.[inputMode || "text"]?.settings_panel ||
    schema?.settings_panel ||
    [];

  // If no schema is provided, show a message (fallback for backwards compatibility)
  if (!schema) {
    return (
      <Card className={`h-full flex flex-col ${className}`}>
        <CardHeader className="flex-shrink-0">
          <CardTitle className="text-base font-medium">Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6 overflow-y-auto flex-1">
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Pipeline ID</h3>
            <Select
              value={pipelineId}
              onValueChange={handlePipelineIdChange}
              disabled={isStreaming || isLoading}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a pipeline" />
              </SelectTrigger>
              <SelectContent>
                {pipelines &&
                  Object.entries(pipelines).map(([id]) => (
                    <SelectItem key={id} value={id}>
                      {id}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
          <p className="text-sm text-muted-foreground">
            Schema information not available. Please wait for schemas to load.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Pipeline ID</h3>
          <Select
            value={pipelineId}
            onValueChange={handlePipelineIdChange}
            disabled={isStreaming || isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a pipeline" />
            </SelectTrigger>
            <SelectContent>
              {pipelines &&
                Object.entries(pipelines).map(([id]) => (
                  <SelectItem key={id} value={id}>
                    {id}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>

        {currentPipeline && (
          <Card>
            <CardContent className="p-4 space-y-2">
              <div>
                <h4 className="text-sm font-semibold">
                  {currentPipeline.name}
                </h4>
              </div>

              <div>
                {(currentPipeline.about ||
                  currentPipeline.docsUrl ||
                  currentPipeline.modified) && (
                  <div className="flex items-stretch gap-1 h-6">
                    {currentPipeline.about && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge
                              variant="outline"
                              className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                            >
                              <Info className="h-3.5 w-3.5" />
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">{currentPipeline.about}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    {currentPipeline.modified && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge
                              variant="outline"
                              className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                            >
                              <Hammer className="h-3.5 w-3.5" />
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>
                              This pipeline contains modifications based on the
                              original project.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    {currentPipeline.docsUrl && (
                      <a
                        href={currentPipeline.docsUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block h-full"
                      >
                        <Badge
                          variant="outline"
                          className="hover:bg-accent cursor-pointer h-full flex items-center"
                        >
                          Docs
                        </Badge>
                      </a>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Render settings controls using SettingsPanelRenderer */}
        {settingsPanelConfig.length > 0 && (
          <SettingsPanelRenderer
            settingsPanel={settingsPanelConfig}
            schema={schema}
            pipelines={pipelines}
            pipelineId={pipelineId}
            isStreaming={isStreaming}
            isLoading={isLoading}
            inputMode={inputMode}
            vaceEnabled={vaceEnabled}
            onVaceEnabledChange={onVaceEnabledChange}
            vaceUseInputVideo={vaceUseInputVideo}
            onVaceUseInputVideoChange={onVaceUseInputVideoChange}
            vaceContextScale={vaceContextScale}
            onVaceContextScaleChange={onVaceContextScaleChange}
            loras={loras}
            onLorasChange={onLorasChange}
            loraMergeStrategy={loraMergeStrategy}
            preprocessorIds={preprocessorIds}
            onPreprocessorIdsChange={onPreprocessorIdsChange}
            manageCache={manageCache}
            onManageCacheChange={onManageCacheChange}
            onResetCache={onResetCache}
            kvCacheAttentionBias={kvCacheAttentionBias}
            onKvCacheAttentionBiasChange={onKvCacheAttentionBiasChange}
            denoisingSteps={denoisingSteps}
            onDenoisingStepsChange={onDenoisingStepsChange}
            defaultDenoisingSteps={defaultDenoisingSteps}
            noiseController={noiseController}
            onNoiseControllerChange={onNoiseControllerChange}
            noiseScale={noiseScale}
            onNoiseScaleChange={onNoiseScaleChange}
            spoutAvailable={spoutAvailable}
            spoutSender={spoutSender}
            onSpoutSenderChange={onSpoutSenderChange}
            resolution={resolution}
            onResolutionChange={onResolutionChange}
            seed={seed}
            onSeedChange={onSeedChange}
            vaeType={vaeType}
            onVaeTypeChange={onVaeTypeChange}
            vaeTypes={vaeTypes}
            quantization={quantization}
            onQuantizationChange={onQuantizationChange}
          />
        )}
      </CardContent>
    </Card>
  );
}
