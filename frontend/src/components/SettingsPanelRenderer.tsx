/**
 * SettingsPanelRenderer - Renders settings controls based on schema configuration.
 *
 * This component reads the `settings_panel` configuration from the pipeline schema
 * and renders controls in the specified order. It supports:
 *
 * 1. Special controls (SettingsControlType) - Complex UI that requires custom handling
 * 2. Dynamic controls (field names) - Simple controls inferred from JSON schema
 *
 * Usage:
 * ```tsx
 * <SettingsPanelRenderer
 *   settingsPanel={schema.settings_panel || modeDefaults?.settings_panel}
 *   schema={schema}
 *   // ... all the control props
 * />
 * ```
 */

import type {
  SettingsPanelItem,
  SettingsControlType,
  PipelineSchemaInfo,
} from "../lib/api";
import type {
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  InputMode,
  PipelineInfo,
  VaeType,
} from "../types";
import { DynamicControl } from "./dynamic-controls";
import {
  resolveSchemaProperty,
  inferControlType,
  getParameterDisplayInfo,
} from "../lib/schemaInference";

// Import special control components
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Toggle } from "./ui/toggle";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { SliderWithInput } from "./ui/slider-with-input";
import { LoRAManager } from "./LoRAManager";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import { Info, Minus, Plus, RotateCcw } from "lucide-react";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../lib/utils";
import { useState } from "react";

// Minimum dimension for most pipelines
const DEFAULT_MIN_DIMENSION = 1;

// Check if a string is a special control type
const SPECIAL_CONTROL_TYPES: SettingsControlType[] = [
  "vace",
  "lora",
  "preprocessor",
  "cache_management",
  "denoising_steps",
  "noise_controls",
  "spout_sender",
];

function isSpecialControl(
  item: SettingsPanelItem
): item is SettingsControlType {
  return SPECIAL_CONTROL_TYPES.includes(item as SettingsControlType);
}

export interface SettingsPanelRendererProps {
  // The settings panel configuration (list of items to render in order)
  settingsPanel: SettingsPanelItem[];
  // Pipeline schema info
  schema: PipelineSchemaInfo;
  // All pipelines for preprocessor dropdown
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: string;

  // State flags
  isStreaming: boolean;
  isLoading: boolean;
  inputMode?: InputMode;

  // VACE controls
  vaceEnabled: boolean;
  onVaceEnabledChange?: (enabled: boolean) => void;
  vaceUseInputVideo: boolean;
  onVaceUseInputVideoChange?: (enabled: boolean) => void;
  vaceContextScale: number;
  onVaceContextScaleChange?: (scale: number) => void;

  // LoRA controls
  loras: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  loraMergeStrategy: LoraMergeStrategy;

  // Preprocessor controls
  preprocessorIds: string[];
  onPreprocessorIdsChange?: (ids: string[]) => void;

  // Cache management controls
  manageCache: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  onResetCache?: () => void;
  kvCacheAttentionBias: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;

  // Denoising steps controls
  denoisingSteps: number[];
  onDenoisingStepsChange?: (steps: number[]) => void;
  defaultDenoisingSteps: number[];

  // Noise controls
  noiseController: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  noiseScale: number;
  onNoiseScaleChange?: (scale: number) => void;

  // Spout controls
  spoutAvailable: boolean;
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;

  // Resolution controls
  resolution: { height: number; width: number };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;

  // Seed control
  seed: number;
  onSeedChange?: (seed: number) => void;

  // VAE type controls
  vaeType: VaeType;
  onVaeTypeChange?: (vaeType: VaeType) => void;
  vaeTypes?: string[];

  // Quantization controls
  quantization: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
}

export function SettingsPanelRenderer({
  settingsPanel,
  schema,
  pipelines,
  pipelineId,
  isStreaming,
  isLoading,
  inputMode,
  vaceEnabled,
  onVaceEnabledChange,
  vaceUseInputVideo,
  onVaceUseInputVideoChange,
  vaceContextScale,
  onVaceContextScaleChange,
  loras,
  onLorasChange,
  loraMergeStrategy,
  preprocessorIds,
  onPreprocessorIdsChange,
  manageCache,
  onManageCacheChange,
  onResetCache,
  kvCacheAttentionBias,
  onKvCacheAttentionBiasChange,
  denoisingSteps,
  onDenoisingStepsChange,
  defaultDenoisingSteps,
  noiseController,
  onNoiseControllerChange,
  noiseScale,
  onNoiseScaleChange,
  spoutAvailable,
  spoutSender,
  onSpoutSenderChange,
  resolution,
  onResolutionChange,
  seed,
  onSeedChange,
  vaeType,
  onVaeTypeChange,
  vaeTypes,
  quantization,
  onQuantizationChange,
}: SettingsPanelRendererProps) {
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
  const [seedError, setSeedError] = useState<string | null>(null);

  // Check if resolution needs adjustment
  const scaleFactor = getResolutionScaleFactor(pipelineId);
  const resolutionWarning =
    scaleFactor &&
    (resolution.height % scaleFactor !== 0 ||
      resolution.width % scaleFactor !== 0)
      ? `Resolution will be adjusted to ${adjustResolutionForPipeline(pipelineId, resolution).resolution.width}×${adjustResolutionForPipeline(pipelineId, resolution).resolution.height} when starting the stream (must be divisible by ${scaleFactor})`
      : null;

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const maxValue = 2048;

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
      if (dimension === "height") {
        setHeightError(null);
      } else {
        setWidthError(null);
      }
    }

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
    const currentPipeline = pipelines?.[pipelineId];
    const minValue = currentPipeline?.minDimension ?? DEFAULT_MIN_DIMENSION;
    const newValue = Math.max(minValue, resolution[dimension] - 1);
    handleResolutionChange(dimension, newValue);
  };

  const handleSeedChange = (value: number) => {
    const minValue = 0;
    const maxValue = 2147483647;

    if (value < minValue) {
      setSeedError(`Must be at least ${minValue}`);
    } else if (value > maxValue) {
      setSeedError(`Must be at most ${maxValue}`);
    } else {
      setSeedError(null);
    }

    onSeedChange?.(value);
  };

  const incrementSeed = () => {
    const maxValue = 2147483647;
    const newValue = Math.min(maxValue, seed + 1);
    handleSeedChange(newValue);
  };

  const decrementSeed = () => {
    const minValue = 0;
    const newValue = Math.max(minValue, seed - 1);
    handleSeedChange(newValue);
  };

  // Render a single item from the settings panel
  const renderItem = (item: SettingsPanelItem, index: number) => {
    if (isSpecialControl(item)) {
      return renderSpecialControl(item, index);
    } else {
      return renderDynamicControl(item, index);
    }
  };

  // Render a special control based on type
  const renderSpecialControl = (
    controlType: SettingsControlType,
    index: number
  ) => {
    switch (controlType) {
      case "vace":
        if (!schema.supports_vace) return null;
        return (
          <div key={`special-${index}`} className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="VACE"
                tooltip="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance."
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
                  VACE is incompatible with FP8 quantization. Please disable
                  quantization to use VACE.
                </p>
              </div>
            )}

            {vaceEnabled && (
              <div className="rounded-lg border bg-card p-3 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label="Use Input Video"
                    tooltip="When enabled in Video input mode, the input video is used for VACE conditioning."
                    className="text-xs text-muted-foreground"
                  />
                  <Toggle
                    pressed={vaceUseInputVideo}
                    onPressedChange={onVaceUseInputVideoChange || (() => {})}
                    variant="outline"
                    size="sm"
                    className="h-7"
                    disabled={isStreaming || isLoading || inputMode !== "video"}
                  >
                    {vaceUseInputVideo ? "ON" : "OFF"}
                  </Toggle>
                </div>
                <div className="flex items-center gap-2">
                  <LabelWithTooltip
                    label="Scale:"
                    tooltip="Scaling factor for VACE hint injection."
                    className="text-xs text-muted-foreground w-16"
                  />
                  <div className="flex-1 min-w-0">
                    <SliderWithInput
                      value={vaceContextScaleSlider.localValue}
                      onValueChange={vaceContextScaleSlider.handleValueChange}
                      onValueCommit={vaceContextScaleSlider.handleValueCommit}
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
        );

      case "lora":
        if (!schema.supports_lora) return null;
        return (
          <div key={`special-${index}`} className="space-y-4">
            <LoRAManager
              loras={loras}
              onLorasChange={onLorasChange}
              disabled={isLoading}
              isStreaming={isStreaming}
              loraMergeStrategy={loraMergeStrategy}
            />
          </div>
        );

      case "preprocessor":
        if (!schema.supports_vace) return null;
        return (
          <div key={`special-${index}`} className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.preprocessor.label}
                tooltip={PARAMETER_METADATA.preprocessor.tooltip}
                className="text-sm text-foreground"
              />
              <Select
                value={preprocessorIds.length > 0 ? preprocessorIds[0] : "none"}
                onValueChange={value => {
                  if (value === "none") {
                    onPreprocessorIdsChange?.([]);
                  } else {
                    onPreprocessorIdsChange?.([value]);
                  }
                }}
                disabled={isStreaming || isLoading}
              >
                <SelectTrigger className="w-[140px] h-7">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  {Object.entries(pipelines || {})
                    .filter(([, info]) => {
                      const isPreprocessor =
                        info.usage?.includes("preprocessor") ?? false;
                      if (!isPreprocessor) return false;
                      if (inputMode) {
                        return (
                          info.supportedModes?.includes(inputMode) ?? false
                        );
                      }
                      return true;
                    })
                    .map(([pid]) => (
                      <SelectItem key={pid} value={pid}>
                        {pid}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        );

      case "cache_management":
        if (!schema.supports_cache_management) return null;
        return (
          <div key={`special-${index}`} className="space-y-4">
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between gap-2">
                <LabelWithTooltip
                  label={PARAMETER_METADATA.manageCache.label}
                  tooltip={PARAMETER_METADATA.manageCache.tooltip}
                  className="text-sm text-foreground"
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
                  className="text-sm text-foreground"
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
        );

      case "denoising_steps":
        return (
          <DenoisingStepsSlider
            key={`special-${index}`}
            value={denoisingSteps}
            onChange={onDenoisingStepsChange || (() => {})}
            defaultValues={defaultDenoisingSteps}
            tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
          />
        );

      case "noise_controls":
        return (
          <div key={`special-${index}`} className="space-y-4">
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between gap-2">
                <LabelWithTooltip
                  label={PARAMETER_METADATA.noiseController.label}
                  tooltip={PARAMETER_METADATA.noiseController.tooltip}
                  className="text-sm text-foreground"
                />
                <Toggle
                  pressed={noiseController}
                  onPressedChange={onNoiseControllerChange || (() => {})}
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
              labelClassName="text-sm text-foreground w-20"
              valueFormatter={noiseScaleSlider.formatValue}
              inputParser={v => parseFloat(v) || 0.0}
            />
          </div>
        );

      case "spout_sender":
        if (!spoutAvailable) return null;
        return (
          <div key={`special-${index}`} className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.spoutSender.label}
                tooltip={PARAMETER_METADATA.spoutSender.tooltip}
                className="text-sm text-foreground"
              />
              <Toggle
                pressed={spoutSender?.enabled ?? false}
                onPressedChange={enabled => {
                  onSpoutSenderChange?.({
                    enabled,
                    name: spoutSender?.name ?? "ScopeOut",
                  });
                }}
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
                  label="Sender Name:"
                  tooltip="The name of the sender that will send video to Spout-compatible apps."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={spoutSender?.name ?? "ScopeOut"}
                  onChange={e => {
                    onSpoutSenderChange?.({
                      enabled: spoutSender?.enabled ?? false,
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
        );

      default:
        return null;
    }
  };

  // Render a dynamic control based on field name
  const renderDynamicControl = (fieldName: string, index: number) => {
    const configSchema = schema.config_schema;
    if (!configSchema?.properties) return null;

    const property = configSchema.properties[fieldName];
    if (!property) {
      // Handle special field names that may not be in schema but are standard
      if (fieldName === "height") {
        return renderHeightControl(index);
      }
      if (fieldName === "width") {
        return renderWidthControl(index);
      }
      if (fieldName === "seed") {
        return renderSeedControl(index);
      }
      if (fieldName === "quantization") {
        return renderQuantizationControl(index);
      }
      if (
        fieldName === "kv_cache_attention_bias" &&
        schema.supports_kv_cache_bias
      ) {
        return renderKvCacheBiasControl(index);
      }
      console.warn(
        `[SettingsPanelRenderer] Unknown field name: "${fieldName}"`
      );
      return null;
    }

    // Handle vae_type specially (uses the vaeTypes array from schema)
    if (fieldName === "vae_type") {
      return renderVaeTypeControl(index);
    }

    // For other fields, use dynamic control rendering
    const resolved = resolveSchemaProperty(property, configSchema.$defs);
    const controlType = inferControlType(resolved);
    const displayInfo = getParameterDisplayInfo(fieldName, property);

    if (controlType === "unknown") {
      console.warn(
        `[SettingsPanelRenderer] Unknown control type for field "${fieldName}"`
      );
      return null;
    }

    // Get current value and onChange handler based on field name
    const { value, onChange } = getFieldValueAndHandler(fieldName);
    if (value === undefined || onChange === undefined) {
      console.warn(
        `[SettingsPanelRenderer] No handler for field "${fieldName}"`
      );
      return null;
    }

    return (
      <DynamicControl
        key={`dynamic-${index}`}
        parameter={{
          name: fieldName,
          property: resolved,
          controlType,
          label: displayInfo.label,
          tooltip: displayInfo.tooltip,
        }}
        value={value}
        onChange={(_name, val) => onChange(val)}
        disabled={isStreaming || isLoading}
      />
    );
  };

  // Get value and onChange handler for a field
  const getFieldValueAndHandler = (
    fieldName: string
  ): { value: unknown; onChange: ((value: unknown) => void) | undefined } => {
    // Map field names to their corresponding props
    // This is a simplified version - in practice you'd want more comprehensive mapping
    switch (fieldName) {
      case "height":
        return {
          value: resolution.height,
          onChange: val =>
            onResolutionChange?.({ ...resolution, height: val as number }),
        };
      case "width":
        return {
          value: resolution.width,
          onChange: val =>
            onResolutionChange?.({ ...resolution, width: val as number }),
        };
      case "seed":
        return { value: seed, onChange: val => onSeedChange?.(val as number) };
      case "vae_type":
        return {
          value: vaeType,
          onChange: val => onVaeTypeChange?.(val as VaeType),
        };
      default:
        return { value: undefined, onChange: undefined };
    }
  };

  // Helper to get display info from schema
  const getDisplayInfo = (fieldName: string) => {
    const configSchema = schema.config_schema;
    if (!configSchema?.properties) {
      return { label: fieldName, tooltip: undefined };
    }
    const property = configSchema.properties[fieldName];
    if (!property) {
      return { label: fieldName, tooltip: undefined };
    }
    return getParameterDisplayInfo(fieldName, property);
  };

  // Render height control
  const renderHeightControl = (index: number) => {
    const displayInfo = getDisplayInfo("height");
    return (
      <div key={`height-${index}`} className="space-y-1">
        <div className="flex items-center gap-2">
          <LabelWithTooltip
            label={displayInfo.label}
            tooltip={displayInfo.tooltip}
            className="text-sm text-foreground w-14"
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
                pipelines?.[pipelineId]?.minDimension ?? DEFAULT_MIN_DIMENSION
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
          <p className="text-xs text-red-500 ml-16">{heightError}</p>
        )}
      </div>
    );
  };

  // Render width control
  const renderWidthControl = (index: number) => {
    const displayInfo = getDisplayInfo("width");
    return (
      <div key={`width-${index}`} className="space-y-1">
        <div className="flex items-center gap-2">
          <LabelWithTooltip
            label={displayInfo.label}
            tooltip={displayInfo.tooltip}
            className="text-sm text-foreground w-14"
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
                pipelines?.[pipelineId]?.minDimension ?? DEFAULT_MIN_DIMENSION
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
          <p className="text-xs text-red-500 ml-16">{widthError}</p>
        )}
        {resolutionWarning && (
          <div className="flex items-start gap-1">
            <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
            <p className="text-xs text-amber-600 dark:text-amber-500">
              {resolutionWarning}
            </p>
          </div>
        )}
      </div>
    );
  };

  // Render seed control
  const renderSeedControl = (index: number) => {
    const displayInfo = getDisplayInfo("seed");
    return (
      <div key={`seed-${index}`} className="space-y-1">
        <div className="flex items-center gap-2">
          <LabelWithTooltip
            label={displayInfo.label}
            tooltip={displayInfo.tooltip}
            className="text-sm text-foreground w-14"
          />
          <div
            className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${seedError ? "border-red-500" : ""}`}
          >
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
              onClick={decrementSeed}
              disabled={isStreaming}
            >
              <Minus className="h-3.5 w-3.5" />
            </Button>
            <Input
              type="number"
              value={seed}
              onChange={e => {
                const value = parseInt(e.target.value);
                if (!isNaN(value)) {
                  handleSeedChange(value);
                }
              }}
              disabled={isStreaming}
              className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              min={0}
              max={2147483647}
            />
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
              onClick={incrementSeed}
              disabled={isStreaming}
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
        {seedError && <p className="text-xs text-red-500 ml-16">{seedError}</p>}
      </div>
    );
  };

  // Render VAE type control
  const renderVaeTypeControl = (index: number) => {
    if (!vaeTypes || vaeTypes.length === 0) return null;
    const displayInfo = getDisplayInfo("vae_type");
    return (
      <div key={`vae-${index}`} className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <LabelWithTooltip
            label={displayInfo.label}
            tooltip={displayInfo.tooltip}
            className="text-sm text-foreground"
          />
          <Select
            value={vaeType}
            onValueChange={value => {
              onVaeTypeChange?.(value as VaeType);
            }}
            disabled={isStreaming}
          >
            <SelectTrigger className="w-[140px] h-7">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {vaeTypes.map(type => (
                <SelectItem key={type} value={type}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    );
  };

  // Render quantization control
  const renderQuantizationControl = (index: number) => {
    if (!schema.supports_quantization) return null;
    const displayInfo = getDisplayInfo("quantization");
    return (
      <div key={`quant-${index}`} className="space-y-4">
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label={displayInfo.label}
              tooltip={displayInfo.tooltip}
              className="text-sm text-foreground"
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
                <SelectItem value="fp8_e4m3fn">fp8_e4m3fn (Dynamic)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {vaceEnabled && (
            <p className="text-xs text-muted-foreground">
              Disabled because VACE is enabled. Disable VACE to use FP8
              quantization.
            </p>
          )}
        </div>
      </div>
    );
  };

  // Render KV cache bias control
  const renderKvCacheBiasControl = (index: number) => {
    const displayInfo = getDisplayInfo("kv_cache_attention_bias");
    return (
      <SliderWithInput
        key={`kvcache-${index}`}
        label={displayInfo.label}
        tooltip={displayInfo.tooltip}
        value={kvCacheAttentionBiasSlider.localValue}
        onValueChange={kvCacheAttentionBiasSlider.handleValueChange}
        onValueCommit={kvCacheAttentionBiasSlider.handleValueCommit}
        min={0.01}
        max={1.0}
        step={0.01}
        incrementAmount={0.01}
        labelClassName="text-sm text-foreground w-20"
        valueFormatter={kvCacheAttentionBiasSlider.formatValue}
        inputParser={v => parseFloat(v) || 1.0}
      />
    );
  };

  return (
    <div className="space-y-6">
      {settingsPanel.map((item, index) => renderItem(item, index))}
    </div>
  );
}
