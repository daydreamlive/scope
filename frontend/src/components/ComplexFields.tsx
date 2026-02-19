/**
 * Complex schema-driven settings components: VACE, LoRA, resolution,
 * cache, denoising steps, noise, quantization.
 * Each block is rendered once per schema configuration (deduplicated by "component" or key).
 */

import { Info, Minus, Plus } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Toggle } from "./ui/toggle";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { SliderWithInput } from "./ui/slider-with-input";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { ImageManager } from "./ImageManager";
import { LoRAManager } from "./LoRAManager";
import { RotateCcw } from "lucide-react";
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
  onOpenLoRAsSettings?: () => void;
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
    return (
      <div key="vace" className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <LabelWithTooltip
            label="VACE"
            tooltip="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, you can use reference images for R2V generation. In Video input mode, a separate toggle controls whether the input video is used for VACE conditioning or for latent initialization. Requires pipeline reload to take effect."
            className="text-sm font-medium"
          />
          <Toggle
            pressed={ctx.vaceEnabled ?? false}
            onPressedChange={ctx.onVaceEnabledChange ?? (() => {})}
            variant="outline"
            size="sm"
            className="h-7"
            disabled={(ctx.isStreaming ?? false) || (ctx.isLoading ?? false)}
          >
            {(ctx.vaceEnabled ?? false) ? "ON" : "OFF"}
          </Toggle>
        </div>
        {ctx.vaceEnabled &&
          ctx.quantization !== null &&
          ctx.quantization !== undefined && (
            <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
              <p className="text-xs text-amber-600 dark:text-amber-500">
                VACE is incompatible with FP8 quantization. Please disable
                quantization to use VACE.
              </p>
            </div>
          )}
        {ctx.vaceEnabled && ctx.vaceContextScaleSlider && (
          <div className="rounded-lg border bg-card p-3 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="Use Input Video"
                tooltip="When enabled in Video input mode, the input video is used for VACE conditioning. When disabled, the input video is used for latent initialization instead, allowing you to use reference images while in Video input mode."
                className="text-xs text-muted-foreground"
              />
              <Toggle
                pressed={ctx.vaceUseInputVideo ?? false}
                onPressedChange={ctx.onVaceUseInputVideoChange ?? (() => {})}
                variant="outline"
                size="sm"
                className="h-7"
                disabled={
                  (ctx.isStreaming ?? false) ||
                  (ctx.isLoading ?? false) ||
                  ctx.inputMode !== "video"
                }
              >
                {(ctx.vaceUseInputVideo ?? false) ? "ON" : "OFF"}
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
                  value={ctx.vaceContextScaleSlider.localValue}
                  onValueChange={ctx.vaceContextScaleSlider.handleValueChange}
                  onValueCommit={ctx.vaceContextScaleSlider.handleValueCommit}
                  min={0}
                  max={2}
                  step={0.1}
                  incrementAmount={0.1}
                  valueFormatter={ctx.vaceContextScaleSlider.formatValue}
                  inputParser={v => parseFloat(v) || 1.0}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (component === "lora" && !rendered.has("lora")) {
    rendered.add("lora");
    return (
      <div key="lora" className="space-y-4">
        <LoRAManager
          loras={ctx.loras ?? []}
          onLorasChange={ctx.onLorasChange ?? (() => {})}
          disabled={ctx.isLoading ?? false}
          isStreaming={ctx.isStreaming ?? false}
          loraMergeStrategy={ctx.loraMergeStrategy ?? "permanent_merge"}
          onOpenLoRAsSettings={ctx.onOpenLoRAsSettings}
        />
      </div>
    );
  }

  if (component === "resolution") {
    if (rendered.has("resolution")) return null;
    rendered.add("resolution");
    const minDim = ctx.minDimension;
    const resolution = ctx.resolution;
    const handleRes = (dim: "height" | "width", v: number) =>
      ctx.onResolutionChange?.(dim, v);
    return (
      <div key="resolution" className="space-y-4">
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
                  className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${ctx.heightError ? "border-red-500" : ""}`}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={() => ctx.decrementResolution?.("height")}
                    disabled={ctx.isStreaming}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                  <Input
                    type="number"
                    value={resolution.height}
                    onChange={e => {
                      const v = parseInt(e.target.value);
                      if (!isNaN(v)) handleRes("height", v);
                    }}
                    disabled={ctx.isStreaming}
                    className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    min={minDim}
                    max={2048}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={() => ctx.incrementResolution?.("height")}
                    disabled={ctx.isStreaming}
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
              {ctx.heightError && (
                <p className="text-xs text-red-500 ml-16">{ctx.heightError}</p>
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
                  className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${ctx.widthError ? "border-red-500" : ""}`}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={() => ctx.decrementResolution?.("width")}
                    disabled={ctx.isStreaming}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                  <Input
                    type="number"
                    value={resolution.width}
                    onChange={e => {
                      const v = parseInt(e.target.value);
                      if (!isNaN(v)) handleRes("width", v);
                    }}
                    disabled={ctx.isStreaming}
                    className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    min={minDim}
                    max={2048}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={() => ctx.incrementResolution?.("width")}
                    disabled={ctx.isStreaming}
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
              {ctx.widthError && (
                <p className="text-xs text-red-500 ml-16">{ctx.widthError}</p>
              )}
            </div>
            {ctx.resolutionWarning && (
              <div className="flex items-start gap-1">
                <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
                <p className="text-xs text-amber-600 dark:text-amber-500">
                  {ctx.resolutionWarning}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (component === "cache" && !rendered.has("cache")) {
    rendered.add("cache");
    if (!ctx.supportsCacheManagement) return null;
    return (
      <div key="cache" className="space-y-4">
        <div className="space-y-2">
          <div className="space-y-2 pt-2">
            {ctx.supportsKvCacheBias && ctx.kvCacheAttentionBiasSlider && (
              <SliderWithInput
                label={PARAMETER_METADATA.kvCacheAttentionBias.label}
                tooltip={PARAMETER_METADATA.kvCacheAttentionBias.tooltip}
                value={ctx.kvCacheAttentionBiasSlider.localValue}
                onValueChange={ctx.kvCacheAttentionBiasSlider.handleValueChange}
                onValueCommit={ctx.kvCacheAttentionBiasSlider.handleValueCommit}
                min={0.01}
                max={1.0}
                step={0.01}
                incrementAmount={0.01}
                labelClassName="text-sm font-medium w-20"
                valueFormatter={ctx.kvCacheAttentionBiasSlider.formatValue}
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
                pressed={ctx.manageCache ?? true}
                onPressedChange={ctx.onManageCacheChange ?? (() => {})}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {(ctx.manageCache ?? true) ? "ON" : "OFF"}
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
                onClick={ctx.onResetCache ?? (() => {})}
                disabled={ctx.manageCache}
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
    return (
      <div key="noise" className="space-y-4">
        <div className="space-y-2">
          <div className="space-y-2 pt-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.noiseController.label}
                tooltip={PARAMETER_METADATA.noiseController.tooltip}
                className="text-sm font-medium"
              />
              <Toggle
                pressed={ctx.noiseController ?? true}
                onPressedChange={ctx.onNoiseControllerChange ?? (() => {})}
                disabled={ctx.isStreaming}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {(ctx.noiseController ?? true) ? "ON" : "OFF"}
              </Toggle>
            </div>
          </div>
          {ctx.noiseScaleSlider && (
            <SliderWithInput
              label={PARAMETER_METADATA.noiseScale.label}
              tooltip={PARAMETER_METADATA.noiseScale.tooltip}
              value={ctx.noiseScaleSlider.localValue}
              onValueChange={ctx.noiseScaleSlider.handleValueChange}
              onValueCommit={ctx.noiseScaleSlider.handleValueCommit}
              min={0.0}
              max={1.0}
              step={0.01}
              incrementAmount={0.01}
              disabled={ctx.noiseController}
              labelClassName="text-sm font-medium w-20"
              valueFormatter={ctx.noiseScaleSlider.formatValue}
              inputParser={v => parseFloat(v) || 0.0}
            />
          )}
        </div>
      </div>
    );
  }

  if (component === "quantization" && !rendered.has("quantization")) {
    rendered.add("quantization");
    if (!ctx.supportsQuantization) return null;
    return (
      <div key="quantization" className="space-y-4">
        <div className="space-y-2">
          <div className="space-y-2 pt-2">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.quantization.label}
                tooltip={PARAMETER_METADATA.quantization.tooltip}
                className="text-sm font-medium"
              />
              <Select
                value={ctx.quantization ?? "none"}
                onValueChange={v =>
                  ctx.onQuantizationChange?.(
                    v === "none" ? null : (v as "fp8_e4m3fn")
                  )
                }
                disabled={
                  (ctx.isStreaming ?? false) || (ctx.vaceEnabled ?? false)
                }
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
            {ctx.vaceEnabled && (
              <p className="text-xs text-muted-foreground">
                Disabled because VACE is enabled. Disable VACE to use FP8
                quantization.
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return null;
}
