/**
 * Complex schema-driven settings components: VACE, LoRA, resolution,
 * cache, denoising steps, noise, quantization.
 * Each block is rendered once per schema configuration (deduplicated by "component" or key).
 * Delegates to settings/ sub-components where possible.
 */

import { ImageManager } from "./ImageManager";
import { LoRAManager } from "./LoRAManager";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { VACEControls } from "./settings/VACEControls";
import { CacheControls } from "./settings/CacheControls";
import { NoiseControls } from "./settings/NoiseControls";
import { ResolutionControls } from "./settings/ResolutionControls";
import { QuantizationControls } from "./settings/QuantizationControls";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { PipelineId, LoRAConfig, LoraMergeStrategy } from "../types";
import type { SchemaFieldUI } from "../lib/schemaSettings";

/** Slider state from useLocalSliderValue, passed in from parent */
export interface SliderState {
  localValue: number;
  handleValueChange: (v: number) => void;
  handleValueCommit: (v: number) => void;
  formatValue: (v: number) => number;
}

/** All data and handlers needed to render complex schema fields. Passed from SettingsPanel. */
export interface SchemaComplexFieldContext {
  pipelineId: PipelineId;
  resolution: { height: number; width: number };
  heightError: string | null;
  widthError: string | null;
  resolutionWarning: string | null;
  minDimension: number;
  onResolutionChange?: (dim: "height" | "width", value: number) => void;
  decrementResolution?: (dim: "height" | "width") => void;
  incrementResolution?: (dim: "height" | "width") => void;
  vaceEnabled?: boolean;
  onVaceEnabledChange?: (enabled: boolean) => void;
  vaceUseInputVideo?: boolean;
  onVaceUseInputVideoChange?: (enabled: boolean) => void;
  vaceContextScaleSlider?: SliderState;
  quantization?: "fp8_e4m3fn" | null;
  loras?: LoRAConfig[];
  onLorasChange?: (loras: LoRAConfig[]) => void;
  loraMergeStrategy?: LoraMergeStrategy;
  manageCache?: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  onResetCache?: () => void;
  kvCacheAttentionBiasSlider?: SliderState;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (steps: number[]) => void;
  defaultDenoisingSteps?: number[];
  noiseScaleSlider?: SliderState;
  noiseController?: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  onQuantizationChange?: (q: "fp8_e4m3fn" | null) => void;
  inputMode?: "text" | "video";
  supportsNoiseControls?: boolean;
  supportsQuantization?: boolean;
  supportsCacheManagement?: boolean;
  supportsKvCacheBias?: boolean;
  isStreaming?: boolean;
  isLoading?: boolean;
  isCloudMode?: boolean;
  /** Per-field overrides for schema-driven fields (e.g. image path). */
  schemaFieldOverrides?: Record<string, unknown>;
  onSchemaFieldOverrideChange?: (
    key: string,
    value: unknown,
    isRuntimeParam?: boolean
  ) => void;
}

export interface SchemaComplexFieldProps {
  component: string;
  fieldKey: string;
  rendered: Set<string>;
  context: SchemaComplexFieldContext;
  /** UI metadata for this field (label, is_load_param). Used for image component. */
  ui?: SchemaFieldUI;
}

/**
 * Renders one complex schema field block. Switches on component (and fieldKey for resolution / noise).
 */
export function SchemaComplexField({
  component,
  fieldKey,
  rendered,
  context: ctx,
  ui,
}: SchemaComplexFieldProps): React.ReactNode {
  if (component === "image") {
    const value = ctx.schemaFieldOverrides?.[fieldKey];
    const path = value == null ? null : String(value);
    const isRuntimeParam = ui?.is_load_param === false;
    const disabled =
      ((ctx.isStreaming ?? false) && !isRuntimeParam) ||
      (ctx.isLoading ?? false);
    return (
      <div key={fieldKey} className="space-y-1">
        {ui?.label != null && (
          <span className="text-xs text-muted-foreground">{ui.label}</span>
        )}
        <ImageManager
          images={path ? [path] : []}
          onImagesChange={images =>
            ctx.onSchemaFieldOverrideChange?.(
              fieldKey,
              images[0] ?? null,
              isRuntimeParam
            )
          }
          disabled={disabled}
          maxImages={1}
          hideLabel
        />
      </div>
    );
  }

  if (component === "vace" && !rendered.has("vace")) {
    rendered.add("vace");
    if (!ctx.vaceContextScaleSlider) return null;
    return (
      <VACEControls
        key="vace"
        vaceEnabled={ctx.vaceEnabled ?? false}
        onVaceEnabledChange={ctx.onVaceEnabledChange ?? (() => {})}
        vaceUseInputVideo={ctx.vaceUseInputVideo ?? false}
        onVaceUseInputVideoChange={ctx.onVaceUseInputVideoChange ?? (() => {})}
        vaceContextScaleSlider={ctx.vaceContextScaleSlider}
        quantization={ctx.quantization ?? null}
        inputMode={ctx.inputMode}
        isStreaming={ctx.isStreaming ?? false}
        isLoading={ctx.isLoading ?? false}
      />
    );
  }

  if (component === "lora" && !rendered.has("lora") && !ctx.isCloudMode) {
    rendered.add("lora");
    return (
      <div key="lora" className="space-y-4">
        <LoRAManager
          loras={ctx.loras ?? []}
          onLorasChange={ctx.onLorasChange ?? (() => {})}
          disabled={ctx.isLoading ?? false}
          isStreaming={ctx.isStreaming ?? false}
          loraMergeStrategy={ctx.loraMergeStrategy ?? "permanent_merge"}
        />
      </div>
    );
  }

  if (component === "resolution") {
    if (rendered.has("resolution")) return null;
    rendered.add("resolution");
    return (
      <ResolutionControls
        key="resolution"
        pipelineId={ctx.pipelineId}
        resolution={ctx.resolution}
        minDimension={ctx.minDimension}
        isStreaming={ctx.isStreaming ?? false}
        onChange={ctx.onResolutionChange ?? (() => {})}
      />
    );
  }

  if (component === "cache" && !rendered.has("cache")) {
    rendered.add("cache");
    if (!ctx.supportsCacheManagement) return null;
    return (
      <CacheControls
        key="cache"
        manageCache={ctx.manageCache ?? true}
        onManageCacheChange={ctx.onManageCacheChange ?? (() => {})}
        onResetCache={ctx.onResetCache ?? (() => {})}
        kvCacheAttentionBiasSlider={
          ctx.kvCacheAttentionBiasSlider ?? {
            localValue: 0.3,
            handleValueChange: () => {},
            handleValueCommit: () => {},
            formatValue: v => v,
          }
        }
        supportsKvCacheBias={ctx.supportsKvCacheBias}
      />
    );
  }

  if (component === "denoising_steps" && !rendered.has("denoising_steps")) {
    rendered.add("denoising_steps");
    return (
      <DenoisingStepsSlider
        key="denoising_steps"
        value={ctx.denoisingSteps ?? []}
        onChange={ctx.onDenoisingStepsChange ?? (() => {})}
        defaultValues={ctx.defaultDenoisingSteps ?? [750, 250]}
        tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
      />
    );
  }

  if (component === "noise") {
    if (rendered.has("noise")) return null;
    rendered.add("noise");
    if (ctx.inputMode !== "video" || !ctx.supportsNoiseControls) return null;
    if (!ctx.noiseScaleSlider) return null;
    return (
      <NoiseControls
        key="noise"
        noiseController={ctx.noiseController ?? true}
        onNoiseControllerChange={ctx.onNoiseControllerChange ?? (() => {})}
        noiseScaleSlider={ctx.noiseScaleSlider}
        isStreaming={ctx.isStreaming ?? false}
      />
    );
  }

  if (component === "quantization" && !rendered.has("quantization")) {
    rendered.add("quantization");
    if (!ctx.supportsQuantization) return null;
    return (
      <QuantizationControls
        key="quantization"
        quantization={ctx.quantization ?? null}
        onQuantizationChange={ctx.onQuantizationChange ?? (() => {})}
        vaceEnabled={ctx.vaceEnabled ?? false}
        isStreaming={ctx.isStreaming ?? false}
      />
    );
  }

  return null;
}
