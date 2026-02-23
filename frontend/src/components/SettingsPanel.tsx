import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
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
import { Badge } from "./ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Toggle } from "./ui/toggle";
import { SliderWithInput } from "./ui/slider-with-input";
import {
  ChevronDown,
  ChevronUp,
  Hammer,
  Info,
  Minus,
  Plus,
  RotateCcw,
  X,
} from "lucide-react";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../lib/utils";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type {
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  InputMode,
  PipelineInfo,
} from "../types";
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

// Minimum dimension for most pipelines (will be overridden by pipeline-specific minDimension from schema)
const DEFAULT_MIN_DIMENSION = 1;

/** Renders primitive config fields from a plugin pipeline's configSchema. */
function PluginConfigFields({
  configSchema,
  inputMode,
  overrides,
  onChange,
  isStreaming = false,
  isLoading = false,
}: {
  configSchema: import("../lib/schemaSettings").ConfigSchemaLike;
  inputMode?: InputMode;
  overrides?: Record<string, unknown>;
  onChange?: (key: string, value: unknown, isRuntimeParam?: boolean) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
}) {
  const fields = parseConfigurationFields(configSchema, inputMode);
  const primitiveFields = fields.filter(
    ({ fieldType }) =>
      !(COMPLEX_COMPONENTS as readonly string[]).includes(fieldType) &&
      fieldType !== "resolution"
  );
  if (primitiveFields.length === 0) return null;

  const enumValuesByRef: Record<string, string[]> = {};
  if (configSchema.$defs) {
    for (const [defName, def] of Object.entries(configSchema.$defs)) {
      if (def?.enum && Array.isArray(def.enum)) {
        enumValuesByRef[defName] = def.enum as string[];
      }
    }
  }

  return (
    <div className="rounded-lg border bg-card p-3 space-y-3">
      {primitiveFields.map(({ key, prop, ui, fieldType }) => {
        const value = overrides?.[key] ?? prop.default;
        const isRuntimeParam = ui.is_load_param === false;
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
            onChange={v => onChange?.(key, v, isRuntimeParam)}
            disabled={(isStreaming && !isRuntimeParam) || isLoading}
            label={ui.label}
            fieldType={
              typeof fieldType === "string" &&
              !(COMPLEX_COMPONENTS as readonly string[]).includes(fieldType)
                ? (fieldType as import("../lib/schemaSettings").PrimitiveFieldType)
                : undefined
            }
            enumValues={enumValues}
          />
        );
      })}
    </div>
  );
}

/** Reusable list UI for selecting, reordering, and configuring multiple pre/postprocessors. */
function ProcessorListSection({
  label,
  tooltip,
  selectedIds,
  onIdsChange,
  availablePipelines,
  allPipelines,
  overrides,
  onOverrideChange,
  inputMode,
  isStreaming = false,
  isLoading = false,
}: {
  label: string;
  tooltip: string;
  selectedIds: string[];
  onIdsChange?: (ids: string[]) => void;
  availablePipelines: string[];
  allPipelines: Record<string, PipelineInfo> | null;
  overrides?: Record<string, Record<string, unknown>>;
  onOverrideChange?: (
    processorId: string,
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
  inputMode?: InputMode;
  isStreaming?: boolean;
  isLoading?: boolean;
}) {
  const disabled = isStreaming || isLoading;
  const unselected = availablePipelines.filter(
    pid => !selectedIds.includes(pid)
  );

  const moveUp = (index: number) => {
    if (index <= 0) return;
    const next = [...selectedIds];
    [next[index - 1], next[index]] = [next[index], next[index - 1]];
    onIdsChange?.(next);
  };

  const moveDown = (index: number) => {
    if (index >= selectedIds.length - 1) return;
    const next = [...selectedIds];
    [next[index], next[index + 1]] = [next[index + 1], next[index]];
    onIdsChange?.(next);
  };

  const remove = (index: number) => {
    const next = selectedIds.filter((_, i) => i !== index);
    onIdsChange?.(next);
  };

  const add = (pid: string) => {
    onIdsChange?.([...selectedIds, pid]);
  };

  return (
    <div className="space-y-2">
      <LabelWithTooltip
        label={label}
        tooltip={tooltip}
        className="text-sm font-medium"
      />

      {selectedIds.map((pid, index) => {
        const configSchema = allPipelines?.[pid]?.configSchema as
          | import("../lib/schemaSettings").ConfigSchemaLike
          | undefined;
        const processorOverrides = overrides?.[pid];

        return (
          <div key={pid} className="rounded-lg border bg-card p-2 space-y-2">
            <div className="flex items-center gap-1">
              <span className="text-xs font-medium flex-1 truncate">{pid}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-5 w-5 shrink-0"
                onClick={() => moveUp(index)}
                disabled={disabled || index === 0}
              >
                <ChevronUp className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-5 w-5 shrink-0"
                onClick={() => moveDown(index)}
                disabled={disabled || index === selectedIds.length - 1}
              >
                <ChevronDown className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-5 w-5 shrink-0"
                onClick={() => remove(index)}
                disabled={disabled}
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
            {configSchema && (
              <PluginConfigFields
                configSchema={configSchema}
                inputMode={inputMode}
                overrides={processorOverrides}
                onChange={
                  onOverrideChange
                    ? (key, value, isRuntimeParam) =>
                        onOverrideChange(pid, key, value, isRuntimeParam)
                    : undefined
                }
                isStreaming={isStreaming}
                isLoading={isLoading}
              />
            )}
          </div>
        );
      })}

      {unselected.length > 0 ? (
        <Select value="" onValueChange={add} disabled={disabled}>
          <SelectTrigger className="w-full h-7">
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Plus className="h-3 w-3" />
              <span>Add</span>
            </div>
          </SelectTrigger>
          <SelectContent>
            {unselected.map(pid => (
              <SelectItem key={pid} value={pid}>
                {pid}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Select value="none" disabled>
          <SelectTrigger className="w-full h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
          </SelectContent>
        </Select>
      )}
    </div>
  );
}

interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  // Resolution is required - parent should always provide from schema defaults
  resolution: {
    height: number;
    width: number;
  };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;
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
  // Whether this pipeline supports noise controls in video mode (schema-derived)
  supportsNoiseControls?: boolean;
  // Output sinks
  outputSinks?: SettingsState["outputSinks"];
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;
  spoutAvailable?: boolean;
  ndiAvailable?: boolean;
  // VACE settings
  vaceEnabled?: boolean;
  onVaceEnabledChange?: (enabled: boolean) => void;
  vaceUseInputVideo?: boolean;
  onVaceUseInputVideoChange?: (enabled: boolean) => void;
  vaceContextScale?: number;
  onVaceContextScaleChange?: (scale: number) => void;
  // Preprocessors
  preprocessorIds?: string[];
  onPreprocessorIdsChange?: (ids: string[]) => void;
  // Postprocessors
  postprocessorIds?: string[];
  onPostprocessorIdsChange?: (ids: string[]) => void;
  // Dynamic schema-driven primitive fields (key = schema field name)
  schemaFieldOverrides?: Record<string, unknown>;
  onSchemaFieldOverrideChange?: (
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
  // Per-processor schema overrides (outer key = pipeline ID, inner = field overrides)
  preprocessorSchemaFieldOverrides?: Record<string, Record<string, unknown>>;
  postprocessorSchemaFieldOverrides?: Record<string, Record<string, unknown>>;
  onPreprocessorSchemaFieldOverrideChange?: (
    processorId: string,
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
  onPostprocessorSchemaFieldOverrideChange?: (
    processorId: string,
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
  isCloudMode?: boolean;
  onOpenLoRAsSettings?: () => void;
}

export function SettingsPanel({
  className = "",
  pipelines,
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  isLoading = false,
  resolution,
  onResolutionChange,
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
  supportsNoiseControls = false,
  outputSinks,
  onOutputSinkChange,
  spoutAvailable = false,
  ndiAvailable = false,
  vaceEnabled = true,
  onVaceEnabledChange,
  vaceUseInputVideo = true,
  onVaceUseInputVideoChange,
  vaceContextScale = 1.0,
  onVaceContextScaleChange,
  preprocessorIds = [],
  onPreprocessorIdsChange,
  postprocessorIds = [],
  onPostprocessorIdsChange,
  schemaFieldOverrides,
  onSchemaFieldOverrideChange,
  preprocessorSchemaFieldOverrides,
  postprocessorSchemaFieldOverrides,
  onPreprocessorSchemaFieldOverrideChange,
  onPostprocessorSchemaFieldOverrideChange,
  isCloudMode = false,
  onOpenLoRAsSettings,
}: SettingsPanelProps) {
  // Local slider state management hooks
  const noiseScaleSlider = useLocalSliderValue(noiseScale, onNoiseScaleChange);
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    onKvCacheAttentionBiasChange
  );
  const vaceContextScaleSlider = useLocalSliderValue(
    vaceContextScale,
    onVaceContextScaleChange
  );

  // Validation error states
  const [heightError, setHeightError] = useState<string | null>(null);
  const [widthError, setWidthError] = useState<string | null>(null);

  // Check if resolution needs adjustment
  const scaleFactor = getResolutionScaleFactor(pipelineId);
  const resolutionWarning =
    scaleFactor &&
    (resolution.height % scaleFactor !== 0 ||
      resolution.width % scaleFactor !== 0)
      ? `Resolution will be adjusted to ${adjustResolutionForPipeline(pipelineId, resolution).resolution.width}×${adjustResolutionForPipeline(pipelineId, resolution).resolution.height} when starting the stream (must be divisible by ${scaleFactor})`
      : null;

  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    // Get min dimension from pipeline schema, fallback to default
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const maxValue = 2048;

    // Validate and set error state
    if (value < minValue) {
      if (dimension === "height") {
        setHeightError(`Must be at least ${minValue}`);
      } else {
        setWidthError(`Must be at least ${minValue}`);
      }
    } else if (value > maxValue) {
      if (dimension === "height") {
        setHeightError(`Must be at most ${maxValue}`);
      } else {
        setWidthError(`Must be at most ${maxValue}`);
      }
    } else {
      // Clear error if valid
      if (dimension === "height") {
        setHeightError(null);
      } else {
        setWidthError(null);
      }
    }

    // Always update the value (even if invalid)
    onResolutionChange?.({
      ...resolution,
      [dimension]: value,
    });
  };

  const incrementResolution = (dimension: "height" | "width") => {
    const maxValue = 2048;
    const newValue = Math.min(maxValue, resolution[dimension] + 1);
    handleResolutionChange(dimension, newValue);
  };

  const decrementResolution = (dimension: "height" | "width") => {
    // Get min dimension from pipeline schema, fallback to default
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const newValue = Math.max(minValue, resolution[dimension] - 1);
    handleResolutionChange(dimension, newValue);
  };

  const currentPipeline = pipelines?.[pipelineId];

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

        {currentPipeline && (
          <Card>
            <CardContent className="p-4 space-y-2">
              <div>
                <h4 className="text-sm font-semibold">
                  {currentPipeline.name}
                  {currentPipeline.pluginName && (
                    <span className="font-normal text-muted-foreground">
                      {" "}
                      ({currentPipeline.pluginName})
                    </span>
                  )}
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

        {/* Preprocessor List */}
        <ProcessorListSection
          label={PARAMETER_METADATA.preprocessor.label}
          tooltip={PARAMETER_METADATA.preprocessor.tooltip}
          selectedIds={preprocessorIds}
          onIdsChange={onPreprocessorIdsChange}
          availablePipelines={Object.entries(pipelines || {})
            .filter(([, info]) => {
              const isPreprocessor =
                info.usage?.includes("preprocessor") ?? false;
              if (!isPreprocessor) return false;
              if (inputMode) {
                return info.supportedModes?.includes(inputMode) ?? false;
              }
              return true;
            })
            .map(([pid]) => pid)}
          allPipelines={pipelines}
          overrides={preprocessorSchemaFieldOverrides}
          onOverrideChange={onPreprocessorSchemaFieldOverrideChange}
          inputMode={inputMode}
          isStreaming={isStreaming}
          isLoading={isLoading}
        />

        {/* Postprocessor List */}
        <ProcessorListSection
          label={PARAMETER_METADATA.postprocessor.label}
          tooltip={PARAMETER_METADATA.postprocessor.tooltip}
          selectedIds={postprocessorIds}
          onIdsChange={onPostprocessorIdsChange}
          availablePipelines={Object.entries(pipelines || {})
            .filter(([, info]) => {
              const isPostprocessor =
                info.usage?.includes("postprocessor") ?? false;
              if (!isPostprocessor) return false;
              return info.supportedModes?.includes("video") ?? false;
            })
            .map(([pid]) => pid)}
          allPipelines={pipelines}
          overrides={postprocessorSchemaFieldOverrides}
          onOverrideChange={onPostprocessorSchemaFieldOverrideChange}
          inputMode={inputMode}
          isStreaming={isStreaming}
          isLoading={isLoading}
        />

        {/* Schema-driven configuration (when configSchema has ui.category===configuration) or legacy */}
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
              heightError,
              widthError,
              resolutionWarning,
              minDimension:
                currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION,
              onResolutionChange: handleResolutionChange,
              decrementResolution,
              incrementResolution,
              vaceEnabled,
              onVaceEnabledChange,
              vaceUseInputVideo,
              onVaceUseInputVideoChange,
              vaceContextScaleSlider,
              quantization: quantization ?? null,
              loras,
              onLorasChange,
              loraMergeStrategy,
              manageCache,
              onManageCacheChange,
              onResetCache,
              kvCacheAttentionBiasSlider,
              denoisingSteps,
              onDenoisingStepsChange,
              defaultDenoisingSteps,
              noiseScaleSlider,
              noiseController,
              onNoiseControllerChange,
              onQuantizationChange,
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
              onSchemaFieldOverrideChange,
              onOpenLoRAsSettings,
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
                      onSchemaFieldOverrideChange?.(key, v, isRuntimeParam);
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
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <LabelWithTooltip
                      label="VACE"
                      tooltip="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, you can use reference images for R2V generation. In Video input mode, a separate toggle controls whether the input video is used for VACE conditioning or for latent initialization. Requires pipeline reload to take effect."
                      className="text-sm font-medium"
                    />
                    <Toggle
                      pressed={vaceEnabled}
                      onPressedChange={onVaceEnabledChange || (() => {})}
                      variant="outline"
                      size="sm"
                      className="h-7"
                      disabled={isStreaming || isLoading}
                    >
                      {vaceEnabled ? "ON" : "OFF"}
                    </Toggle>
                  </div>
                  {vaceEnabled && quantization !== null && (
                    <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
                      <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
                      <p className="text-xs text-amber-600 dark:text-amber-500">
                        VACE is incompatible with FP8 quantization. Please
                        disable quantization to use VACE.
                      </p>
                    </div>
                  )}
                  {vaceEnabled && (
                    <div className="rounded-lg border bg-card p-3 space-y-3">
                      <div className="flex items-center justify-between gap-2">
                        <LabelWithTooltip
                          label="Use Input Video"
                          tooltip="When enabled in Video input mode, the input video is used for VACE conditioning. When disabled, the input video is used for latent initialization instead, allowing you to use reference images while in Video input mode."
                          className="text-xs text-muted-foreground"
                        />
                        <Toggle
                          pressed={vaceUseInputVideo}
                          onPressedChange={
                            onVaceUseInputVideoChange || (() => {})
                          }
                          variant="outline"
                          size="sm"
                          className="h-7"
                          disabled={
                            isStreaming || isLoading || inputMode !== "video"
                          }
                        >
                          {vaceUseInputVideo ? "ON" : "OFF"}
                        </Toggle>
                      </div>
                      <div className="flex items-center gap-2">
                        <LabelWithTooltip
                          label="Scale"
                          tooltip="Scaling factor for VACE hint injection. Higher values make reference images more influential."
                          className="text-xs text-muted-foreground w-16"
                        />
                        <div className="flex-1 min-w-0">
                          <SliderWithInput
                            value={vaceContextScaleSlider.localValue}
                            onValueChange={
                              vaceContextScaleSlider.handleValueChange
                            }
                            onValueCommit={
                              vaceContextScaleSlider.handleValueCommit
                            }
                            min={0}
                            max={2}
                            step={0.1}
                            incrementAmount={0.1}
                            valueFormatter={vaceContextScaleSlider.formatValue}
                            inputParser={v => parseFloat(v) || 1.0}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {currentPipeline?.supportsLoRA && (
                <div className="space-y-4">
                  <LoRAManager
                    loras={loras}
                    onLorasChange={onLorasChange}
                    disabled={isLoading}
                    isStreaming={isStreaming}
                    loraMergeStrategy={loraMergeStrategy}
                    onOpenLoRAsSettings={onOpenLoRAsSettings}
                  />
                </div>
              )}

              {/* Resolution controls - shown for pipelines that support quantization (implies they need resolution config) */}
              {pipelines?.[pipelineId]?.supportsQuantization && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <LabelWithTooltip
                            label={PARAMETER_METADATA.height.label}
                            tooltip={PARAMETER_METADATA.height.tooltip}
                            className="text-sm font-medium w-14"
                          />
                          <div
                            className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${heightError ? "border-red-500" : ""}`}
                          >
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                              onClick={() => decrementResolution("height")}
                              disabled={isStreaming}
                            >
                              <Minus className="h-3.5 w-3.5" />
                            </Button>
                            <Input
                              type="number"
                              value={resolution.height}
                              onChange={e => {
                                const value = parseInt(e.target.value);
                                if (!isNaN(value)) {
                                  handleResolutionChange("height", value);
                                }
                              }}
                              disabled={isStreaming}
                              className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                              min={
                                pipelines?.[pipelineId]?.minDimension ??
                                DEFAULT_MIN_DIMENSION
                              }
                              max={2048}
                            />
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                              onClick={() => incrementResolution("height")}
                              disabled={isStreaming}
                            >
                              <Plus className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </div>
                        {heightError && (
                          <p className="text-xs text-red-500 ml-16">
                            {heightError}
                          </p>
                        )}
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <LabelWithTooltip
                            label={PARAMETER_METADATA.width.label}
                            tooltip={PARAMETER_METADATA.width.tooltip}
                            className="text-sm font-medium w-14"
                          />
                          <div
                            className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${widthError ? "border-red-500" : ""}`}
                          >
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                              onClick={() => decrementResolution("width")}
                              disabled={isStreaming}
                            >
                              <Minus className="h-3.5 w-3.5" />
                            </Button>
                            <Input
                              type="number"
                              value={resolution.width}
                              onChange={e => {
                                const value = parseInt(e.target.value);
                                if (!isNaN(value)) {
                                  handleResolutionChange("width", value);
                                }
                              }}
                              disabled={isStreaming}
                              className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                              min={
                                pipelines?.[pipelineId]?.minDimension ??
                                DEFAULT_MIN_DIMENSION
                              }
                              max={2048}
                            />
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                              onClick={() => incrementResolution("width")}
                              disabled={isStreaming}
                            >
                              <Plus className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </div>
                        {widthError && (
                          <p className="text-xs text-red-500 ml-16">
                            {widthError}
                          </p>
                        )}
                      </div>
                      {resolutionWarning && (
                        <div className="flex items-start gap-1">
                          <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
                          <p className="text-xs text-amber-600 dark:text-amber-500">
                            {resolutionWarning}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Cache management controls - shown for pipelines that support it */}
              {pipelines?.[pipelineId]?.supportsCacheManagement && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="space-y-2 pt-2">
                      {/* KV Cache bias control - shown for pipelines that support it */}
                      {pipelines?.[pipelineId]?.supportsKvCacheBias && (
                        <SliderWithInput
                          label={PARAMETER_METADATA.kvCacheAttentionBias.label}
                          tooltip={
                            PARAMETER_METADATA.kvCacheAttentionBias.tooltip
                          }
                          value={kvCacheAttentionBiasSlider.localValue}
                          onValueChange={
                            kvCacheAttentionBiasSlider.handleValueChange
                          }
                          onValueCommit={
                            kvCacheAttentionBiasSlider.handleValueCommit
                          }
                          min={0.01}
                          max={1.0}
                          step={0.01}
                          incrementAmount={0.01}
                          labelClassName="text-sm font-medium w-20"
                          valueFormatter={
                            kvCacheAttentionBiasSlider.formatValue
                          }
                          inputParser={v => parseFloat(v) || 1.0}
                        />
                      )}

                      <div className="flex items-center justify-between gap-2">
                        <LabelWithTooltip
                          label={PARAMETER_METADATA.manageCache.label}
                          tooltip={PARAMETER_METADATA.manageCache.tooltip}
                          className="text-sm font-medium"
                        />
                        <Toggle
                          pressed={manageCache}
                          onPressedChange={onManageCacheChange || (() => {})}
                          variant="outline"
                          size="sm"
                          className="h-7"
                        >
                          {manageCache ? "ON" : "OFF"}
                        </Toggle>
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <LabelWithTooltip
                          label={PARAMETER_METADATA.resetCache.label}
                          tooltip={PARAMETER_METADATA.resetCache.tooltip}
                          className="text-sm font-medium"
                        />
                        <Button
                          type="button"
                          onClick={onResetCache || (() => {})}
                          disabled={manageCache}
                          variant="outline"
                          size="sm"
                          className="h-7 w-7 p-0"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Denoising steps - shown for pipelines that support quantization (implies advanced diffusion features) */}
              {pipelines?.[pipelineId]?.supportsQuantization && (
                <DenoisingStepsSlider
                  value={denoisingSteps}
                  onChange={onDenoisingStepsChange || (() => {})}
                  defaultValues={defaultDenoisingSteps}
                  tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
                />
              )}

              {/* Noise controls - show for video mode on supported pipelines (schema-derived) */}
              {inputMode === "video" && supportsNoiseControls && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="space-y-2 pt-2">
                      <div className="flex items-center justify-between gap-2">
                        <LabelWithTooltip
                          label={PARAMETER_METADATA.noiseController.label}
                          tooltip={PARAMETER_METADATA.noiseController.tooltip}
                          className="text-sm font-medium"
                        />
                        <Toggle
                          pressed={noiseController}
                          onPressedChange={
                            onNoiseControllerChange || (() => {})
                          }
                          disabled={isStreaming}
                          variant="outline"
                          size="sm"
                          className="h-7"
                        >
                          {noiseController ? "ON" : "OFF"}
                        </Toggle>
                      </div>
                    </div>

                    <SliderWithInput
                      label={PARAMETER_METADATA.noiseScale.label}
                      tooltip={PARAMETER_METADATA.noiseScale.tooltip}
                      value={noiseScaleSlider.localValue}
                      onValueChange={noiseScaleSlider.handleValueChange}
                      onValueCommit={noiseScaleSlider.handleValueCommit}
                      min={0.0}
                      max={1.0}
                      step={0.01}
                      incrementAmount={0.01}
                      disabled={noiseController}
                      labelClassName="text-sm font-medium w-20"
                      valueFormatter={noiseScaleSlider.formatValue}
                      inputParser={v => parseFloat(v) || 0.0}
                    />
                  </div>
                </div>
              )}

              {/* Quantization controls - shown for pipelines that support it */}
              {pipelines?.[pipelineId]?.supportsQuantization && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="space-y-2 pt-2">
                      <div className="flex items-center justify-between gap-2">
                        <LabelWithTooltip
                          label={PARAMETER_METADATA.quantization.label}
                          tooltip={PARAMETER_METADATA.quantization.tooltip}
                          className="text-sm font-medium"
                        />
                        <Select
                          value={quantization || "none"}
                          onValueChange={value => {
                            onQuantizationChange?.(
                              value === "none" ? null : (value as "fp8_e4m3fn")
                            );
                          }}
                          disabled={isStreaming || vaceEnabled}
                        >
                          <SelectTrigger className="w-[140px] h-7">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            <SelectItem value="fp8_e4m3fn">
                              fp8_e4m3fn (Dynamic)
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      {/* Note when quantization is disabled due to VACE */}
                      {vaceEnabled && (
                        <p className="text-xs text-muted-foreground">
                          Disabled because VACE is enabled. Disable VACE to use
                          FP8 quantization.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </>
          );
        })()}

        {/* Spout Output */}
        {spoutAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="Spout Output"
                tooltip="Send video to Spout-compatible apps (Windows) like TouchDesigner, Resolume, OBS."
                className="text-sm font-medium"
              />
              <Toggle
                pressed={outputSinks?.spout?.enabled ?? false}
                onPressedChange={enabled => {
                  onOutputSinkChange?.("spout", {
                    enabled,
                    name: outputSinks?.spout?.name ?? "ScopeOut",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {outputSinks?.spout?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {outputSinks?.spout?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name"
                  tooltip="The name visible to Spout receivers."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={outputSinks?.spout?.name ?? "ScopeOut"}
                  onChange={e => {
                    onOutputSinkChange?.("spout", {
                      enabled: outputSinks?.spout?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="ScopeOut"
                />
              </div>
            )}
          </div>
        )}

        {/* NDI Output */}
        {ndiAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.ndiSender.label}
                tooltip={PARAMETER_METADATA.ndiSender.tooltip}
                className="text-sm font-medium"
              />
              <Toggle
                pressed={outputSinks?.ndi?.enabled ?? false}
                onPressedChange={enabled => {
                  onOutputSinkChange?.("ndi", {
                    enabled,
                    name: outputSinks?.ndi?.name ?? "Scope",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {outputSinks?.ndi?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {outputSinks?.ndi?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name"
                  tooltip="The name visible to NDI receivers on the network."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={outputSinks?.ndi?.name ?? "Scope"}
                  onChange={e => {
                    onOutputSinkChange?.("ndi", {
                      enabled: outputSinks?.ndi?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="Scope"
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
