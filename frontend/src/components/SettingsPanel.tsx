import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type { PipelineId } from "../types";
import { LoRAManager } from "./LoRAManager";
import {
  parseConfigurationFields,
  COMPLEX_COMPONENTS,
} from "../lib/schemaSettings";
import {
  SchemaComplexField,
  type SchemaComplexFieldContext,
} from "./ComplexFields";
import { SchemaPrimitiveField } from "./PrimitiveFields";
import { useAppStore } from "../stores";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamContext } from "../contexts/StreamContext";
import { PipelineInfoCard } from "./settings/PipelineInfoCard";
import { ProcessorSelector } from "./settings/ProcessorSelector";
import { ResolutionControls } from "./settings/ResolutionControls";
import { VACEControls } from "./settings/VACEControls";
import { CacheControls } from "./settings/CacheControls";
import { NoiseControls } from "./settings/NoiseControls";
import { SpoutSenderSettings } from "./settings/SpoutSenderSettings";
import { QuantizationControls } from "./settings/QuantizationControls";

// Minimum dimension for most pipelines (will be overridden by pipeline-specific minDimension from schema)
const DEFAULT_MIN_DIMENSION = 1;

interface SettingsPanelProps {
  className?: string;
}

export function SettingsPanel({ className = "" }: SettingsPanelProps) {
  // Read from store
  const settings = useAppStore(s => s.settings);

  // Read from contexts
  const { pipelines } = usePipelinesContext();
  const {
    actions,
    updateSettings,
    isStreaming,
    isLoading,
    isCloudMode,
    spoutAvailable,
    getDefaults,
  } = useStreamContext();

  // Derive values with fallbacks (previously computed in StreamPage)
  const pipelineId = settings.pipelineId;
  const defaults = getDefaults(pipelineId, settings.inputMode);
  const resolution = settings.resolution || {
    height: defaults.height,
    width: defaults.width,
  };
  const denoisingSteps = settings.denoisingSteps ||
    defaults.denoisingSteps || [750, 250];
  const defaultDenoisingSteps = defaults.denoisingSteps || [750, 250];
  const noiseScale = settings.noiseScale ?? 0.7;
  const noiseController = settings.noiseController ?? true;
  const manageCache = settings.manageCache ?? true;
  const quantization =
    settings.quantization !== undefined ? settings.quantization : "fp8_e4m3fn";
  const kvCacheAttentionBias = settings.kvCacheAttentionBias ?? 0.3;
  const loras = settings.loras || [];
  const loraMergeStrategy = settings.loraMergeStrategy ?? "permanent_merge";
  const inputMode = settings.inputMode;
  const supportsNoiseControls = actions.supportsNoiseControls(pipelineId);
  const spoutSender = settings.spoutSender;
  const vaceEnabled =
    settings.vaceEnabled ??
    (pipelines?.[pipelineId]?.supportsVACE && inputMode !== "video") ??
    false;
  const vaceUseInputVideo = settings.vaceUseInputVideo ?? false;
  const vaceContextScale = settings.vaceContextScale ?? 1.0;
  const preprocessorIds = settings.preprocessorIds ?? [];
  const postprocessorIds = settings.postprocessorIds ?? [];
  const schemaFieldOverrides = settings.schemaFieldOverrides ?? {};

  const noiseScaleSlider = useLocalSliderValue(
    noiseScale,
    actions.handleNoiseScaleChange
  );
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    actions.handleKvCacheAttentionBiasChange
  );
  const vaceContextScaleSlider = useLocalSliderValue(
    vaceContextScale,
    actions.handleVaceContextScaleChange
  );

  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      actions.handlePipelineIdChange(value as PipelineId);
    }
  };

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    updateSettings({
      resolution: {
        ...resolution,
        [dimension]: value,
      },
    });
  };

  const currentPipeline = pipelines?.[pipelineId];

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        {/* Pipeline Selector */}
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
                (() => {
                  const entries = Object.entries(pipelines);
                  const builtIn = entries.filter(
                    ([, info]) => !info.pluginName
                  );
                  const plugin = entries.filter(([, info]) => info.pluginName);
                  return (
                    <>
                      {builtIn.length > 0 && (
                        <SelectGroup>
                          <SelectLabel className="text-xs text-muted-foreground font-bold">
                            Built-in Pipelines
                          </SelectLabel>
                          {builtIn.map(([id]) => (
                            <SelectItem key={id} value={id}>
                              {id}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      )}
                      {builtIn.length > 0 && plugin.length > 0 && (
                        <SelectSeparator />
                      )}
                      {plugin.length > 0 && (
                        <SelectGroup>
                          <SelectLabel className="text-xs text-muted-foreground font-bold">
                            Plugin Pipelines
                          </SelectLabel>
                          {plugin.map(([id]) => (
                            <SelectItem key={id} value={id}>
                              {id}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      )}
                    </>
                  );
                })()}
            </SelectContent>
          </Select>
        </div>

        {currentPipeline && <PipelineInfoCard pipeline={currentPipeline} />}

        <ProcessorSelector
          type="preprocessor"
          selectedIds={preprocessorIds}
          onChange={actions.handlePreprocessorIdsChange}
          pipelines={pipelines}
          inputMode={inputMode}
          disabled={isStreaming || isLoading}
        />

        <ProcessorSelector
          type="postprocessor"
          selectedIds={postprocessorIds}
          onChange={actions.handlePostprocessorIdsChange}
          pipelines={pipelines}
          disabled={isStreaming || isLoading}
        />

        {/* Schema-driven configuration or legacy controls */}
        {(() => {
          const configSchema = currentPipeline?.configSchema as
            | import("../lib/schemaSettings").ConfigSchemaLike
            | undefined;
          const parsedFields = parseConfigurationFields(
            configSchema,
            inputMode
          );
          const rendered = new Set<string>();

          // Enum values from schema $defs for $ref-based enums
          const enumValuesByRef: Record<string, string[]> = {};
          if (configSchema?.$defs) {
            for (const [defName, def] of Object.entries(configSchema.$defs)) {
              if (def?.enum && Array.isArray(def.enum)) {
                enumValuesByRef[defName] = def.enum as string[];
              }
            }
          }

          if (parsedFields.length > 0) {
            const schemaComplexContext: SchemaComplexFieldContext = {
              pipelineId,
              resolution,
              heightError: null,
              widthError: null,
              resolutionWarning: null,
              minDimension:
                currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION,
              onResolutionChange: handleResolutionChange,
              decrementResolution: () => {},
              incrementResolution: () => {},
              vaceEnabled,
              onVaceEnabledChange: actions.handleVaceEnabledChange,
              vaceUseInputVideo,
              onVaceUseInputVideoChange: actions.handleVaceUseInputVideoChange,
              vaceContextScaleSlider,
              quantization: quantization ?? null,
              loras,
              onLorasChange: actions.handleLorasChange,
              loraMergeStrategy,
              manageCache,
              onManageCacheChange: actions.handleManageCacheChange,
              onResetCache: actions.handleResetCache,
              kvCacheAttentionBiasSlider,
              denoisingSteps,
              onDenoisingStepsChange: actions.handleDenoisingStepsChange,
              defaultDenoisingSteps,
              noiseScaleSlider,
              noiseController,
              onNoiseControllerChange: actions.handleNoiseControllerChange,
              onQuantizationChange: actions.handleQuantizationChange,
              inputMode,
              supportsNoiseControls,
              supportsQuantization:
                pipelines?.[pipelineId]?.supportsQuantization,
              supportsCacheManagement:
                pipelines?.[pipelineId]?.supportsCacheManagement,
              supportsKvCacheBias: pipelines?.[pipelineId]?.supportsKvCacheBias,
              isStreaming,
              isLoading,
              isCloudMode,
              schemaFieldOverrides,
              onSchemaFieldOverrideChange:
                actions.handleSchemaFieldOverrideChange,
            };
            return (
              <>
                {parsedFields
                  .map(({ key, prop, ui, fieldType }) => {
                    const comp = ui.component;
                    const complexNode = SchemaComplexField({
                      component: comp ?? "",
                      fieldKey: key,
                      rendered,
                      context: schemaComplexContext,
                      ui,
                    });
                    if (complexNode != null) return complexNode;
                    if (
                      comp &&
                      (COMPLEX_COMPONENTS as readonly string[]).includes(comp)
                    )
                      return null;
                    // height/width already shown in resolution block – don't render as primitives
                    if (comp === "resolution" || fieldType === "resolution")
                      return null;
                    const value = schemaFieldOverrides?.[key] ?? prop.default;
                    const isRuntimeParam = ui.is_load_param === false;
                    const setValue = (v: unknown) =>
                      actions.handleSchemaFieldOverrideChange(
                        key,
                        v,
                        isRuntimeParam
                      );
                    const primitiveDisabled =
                      (isStreaming && !isRuntimeParam) || isLoading;
                    const enumValues =
                      fieldType === "enum" && typeof prop.$ref === "string"
                        ? enumValuesByRef[prop.$ref.split("/").pop() ?? ""]
                        : undefined;
                    return (
                      <SchemaPrimitiveField
                        key={key}
                        fieldKey={key}
                        prop={prop}
                        value={value}
                        onChange={setValue}
                        disabled={primitiveDisabled}
                        label={ui.label}
                        fieldType={
                          typeof fieldType === "string" &&
                          !(COMPLEX_COMPONENTS as readonly string[]).includes(
                            fieldType
                          )
                            ? (fieldType as import("../lib/schemaSettings").PrimitiveFieldType)
                            : undefined
                        }
                        enumValues={enumValues}
                      />
                    );
                  })
                  .filter(Boolean)}
              </>
            );
          }

          // Legacy: no configSchema ui fields – use supportsVACE, supportsLoRA, etc.
          return (
            <>
              {currentPipeline?.supportsVACE && (
                <VACEControls
                  vaceEnabled={vaceEnabled}
                  onVaceEnabledChange={actions.handleVaceEnabledChange}
                  vaceUseInputVideo={vaceUseInputVideo}
                  onVaceUseInputVideoChange={
                    actions.handleVaceUseInputVideoChange
                  }
                  vaceContextScaleSlider={vaceContextScaleSlider}
                  quantization={quantization ?? null}
                  inputMode={inputMode}
                  isStreaming={isStreaming}
                  isLoading={isLoading}
                />
              )}

              {currentPipeline?.supportsLoRA && !isCloudMode && (
                <div className="space-y-4">
                  <LoRAManager
                    loras={loras}
                    onLorasChange={actions.handleLorasChange}
                    disabled={isLoading}
                    isStreaming={isStreaming}
                    loraMergeStrategy={loraMergeStrategy}
                  />
                </div>
              )}

              {pipelines?.[pipelineId]?.supportsQuantization && (
                <ResolutionControls
                  pipelineId={pipelineId}
                  resolution={resolution}
                  minDimension={
                    currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION
                  }
                  isStreaming={isStreaming}
                  onChange={handleResolutionChange}
                />
              )}

              {pipelines?.[pipelineId]?.supportsCacheManagement && (
                <CacheControls
                  manageCache={manageCache}
                  onManageCacheChange={actions.handleManageCacheChange}
                  onResetCache={actions.handleResetCache}
                  kvCacheAttentionBiasSlider={kvCacheAttentionBiasSlider}
                  supportsKvCacheBias={
                    pipelines?.[pipelineId]?.supportsKvCacheBias
                  }
                />
              )}

              {pipelines?.[pipelineId]?.supportsQuantization && (
                <DenoisingStepsSlider
                  value={denoisingSteps}
                  onChange={actions.handleDenoisingStepsChange}
                  defaultValues={defaultDenoisingSteps}
                  tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
                />
              )}

              {inputMode === "video" && supportsNoiseControls && (
                <NoiseControls
                  noiseController={noiseController}
                  onNoiseControllerChange={actions.handleNoiseControllerChange}
                  noiseScaleSlider={noiseScaleSlider}
                  isStreaming={isStreaming}
                />
              )}

              {pipelines?.[pipelineId]?.supportsQuantization && (
                <QuantizationControls
                  quantization={quantization ?? null}
                  onQuantizationChange={actions.handleQuantizationChange}
                  vaceEnabled={vaceEnabled}
                  isStreaming={isStreaming}
                />
              )}
            </>
          );
        })()}

        {spoutAvailable && (
          <SpoutSenderSettings
            spoutSender={spoutSender}
            isStreaming={isStreaming}
            onChange={actions.handleSpoutSenderChange}
          />
        )}
      </CardContent>
    </Card>
  );
}
