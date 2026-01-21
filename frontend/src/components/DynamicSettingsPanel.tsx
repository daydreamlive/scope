/**
 * DynamicSettingsPanel - Main orchestrator for dynamic parameter controls.
 *
 * This component:
 * - Extracts fields from pipeline schema
 * - Groups fields by category
 * - Renders sections with DynamicSection
 * - Handles uncategorized fields
 * - Integrates with existing special controls (VACE, LoRA, etc.)
 */

import { useMemo } from "react";
import { DynamicSection, type FieldInfo } from "./DynamicSection";
import { extractRenderableParameters } from "../lib/schemaInference";
import { getCategoryConfig } from "../config/categoryConfig";
import type { PipelineSchemaInfo } from "../lib/api";
import type { InputMode } from "../types";
import { LoRAManager } from "./LoRAManager";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Button } from "./ui/button";
import { Info, RotateCcw } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import type {
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  PipelineInfo,
} from "../types";

export interface DynamicSettingsPanelProps {
  /** Pipeline schema info */
  schema: PipelineSchemaInfo;
  /** Current field values */
  fieldValues: Record<string, unknown>;
  /** Current input mode */
  inputMode?: InputMode;
  /** Change handler for field values */
  onFieldChange: (fieldName: string, value: unknown) => void;
  /** Whether controls are disabled */
  disabled?: boolean;
  /** All pipelines for preprocessor dropdown */
  pipelines?: Record<string, PipelineInfo> | null;
  /** Pipeline ID */
  pipelineId?: string;
  /** LoRA controls */
  loras?: LoRAConfig[];
  onLorasChange?: (loras: LoRAConfig[]) => void;
  loraMergeStrategy?: LoraMergeStrategy;
  /** Preprocessor controls */
  preprocessorIds?: string[];
  onPreprocessorIdsChange?: (ids: string[]) => void;
  /** Cache management */
  onResetCache?: () => void;
  /** Spout controls */
  spoutAvailable?: boolean;
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
}

/**
 * Groups fields by category and extracts UI metadata.
 */
function groupFieldsByCategory(
  schema: PipelineSchemaInfo
): Map<string, FieldInfo[]> {
  const categoryMap = new Map<string, FieldInfo[]>();
  const uncategorized: FieldInfo[] = [];

  if (!schema.config_schema?.properties) {
    return categoryMap;
  }

  // Extract all renderable parameters
  const parameters = extractRenderableParameters(schema.config_schema);

  for (const param of parameters) {
    const category =
      param.property.uiMetadata?.["ui:category"] || "uncategorized";
    const order = param.property.uiMetadata?.["ui:order"] ?? 999;

    const fieldInfo: FieldInfo = {
      name: param.name,
      property: param.property,
      controlType: param.controlType,
      label: param.label,
      tooltip: param.tooltip,
      order,
    };

    if (category === "uncategorized") {
      uncategorized.push(fieldInfo);
    } else {
      if (!categoryMap.has(category)) {
        categoryMap.set(category, []);
      }
      categoryMap.get(category)!.push(fieldInfo);
    }
  }

  // Add uncategorized fields to map if any
  if (uncategorized.length > 0) {
    categoryMap.set("uncategorized", uncategorized);
  }

  return categoryMap;
}

/**
 * Gets category order for sorting sections.
 */
function getCategoryOrder(
  categoryName: string,
  backendCategoryConfig?: Record<string, import("../lib/api").CategoryConfig>
): number {
  const config = getCategoryConfig(categoryName, backendCategoryConfig);
  return config.order;
}

/**
 * SpecialControlRenderer - Component for rendering special controls (VACE, LoRA, etc.)
 */
function SpecialControlRenderer({
  categoryName,
  schema,
  fieldValues,
  inputMode,
  disabled,
  props,
}: {
  categoryName: string;
  schema: PipelineSchemaInfo;
  fieldValues: Record<string, unknown>;
  inputMode?: InputMode;
  disabled?: boolean;
  props?: Omit<DynamicSettingsPanelProps, "schema" | "fieldValues" | "inputMode" | "disabled"> & { onFieldChange?: (fieldName: string, value: unknown) => void };
}) {
  switch (categoryName) {
    case "vace":
      // VACE is now fully field-based (vace_enabled, vace_use_input_video, vace_context_scale)
      // No special control needed - all fields are rendered via DynamicField
      // Show warning if VACE enabled with quantization
      if (!schema.supports_vace) return null;
      const vaceEnabled = fieldValues.vace_enabled as boolean | undefined;
      const quantization = fieldValues.quantization as "fp8_e4m3fn" | null;

      if (vaceEnabled && quantization !== null) {
        return (
          <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
            <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600 dark:text-amber-500" />
            <p className="text-xs text-amber-600 dark:text-amber-500">
              VACE is incompatible with FP8 quantization. Please disable
              quantization to use VACE.
            </p>
          </div>
        );
      }
      return null;

    case "lora":
      if (!schema.supports_lora || !props) return null;
      const { loras, onLorasChange, loraMergeStrategy } = props;
      if (!loras || !onLorasChange) return null;
      return (
        <LoRAManager
          loras={loras}
          onLorasChange={onLorasChange}
          disabled={disabled}
          isStreaming={disabled}
          loraMergeStrategy={loraMergeStrategy || "permanent_merge"}
        />
      );

    case "preprocessor":
      if (!schema.supports_vace || !props) return null;
      const { pipelines, pipelineId, preprocessorIds, onPreprocessorIdsChange } = props;
      if (!pipelines || !pipelineId) return null;
      return (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <LabelWithTooltip
              label="Preprocessor:"
              tooltip="Select a preprocessor to apply before the main pipeline."
              className="text-sm text-foreground"
            />
            <Select
              value={preprocessorIds && preprocessorIds.length > 0 ? preprocessorIds[0] : "none"}
              onValueChange={value => {
                if (value === "none") {
                  onPreprocessorIdsChange?.([]);
                } else {
                  onPreprocessorIdsChange?.([value]);
                }
              }}
              disabled={disabled}
            >
              <SelectTrigger className="w-[140px] h-7">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {Object.entries(pipelines)
                  .filter(([, info]) => {
                    const isPreprocessor = info.usage?.includes("preprocessor") ?? false;
                    if (!isPreprocessor) return false;
                    if (inputMode) {
                      return info.supportedModes?.includes(inputMode) ?? false;
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

    case "cache":
      if (!props) return null;
      const { onResetCache } = props;
      const manageCache = fieldValues.manage_cache as boolean | undefined;
      return (
        <div className="space-y-2">
          {onResetCache && (
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="Reset Cache:"
                tooltip="Clears previous frames from cache allowing new frames to be generated with fresh history. Only available when Manage Cache is disabled."
                className="text-sm text-foreground"
              />
              <Button
                type="button"
                onClick={onResetCache}
                disabled={manageCache || disabled}
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
      );

    default:
      return null;
  }
}

export function DynamicSettingsPanel({
  schema,
  fieldValues,
  inputMode,
  onFieldChange,
  disabled = false,
  ...specialControlProps
}: DynamicSettingsPanelProps) {
  // Group fields by category
  const categoryMap = useMemo(
    () => groupFieldsByCategory(schema),
    [schema]
  );

  // Sort categories by order
  const sortedCategories = useMemo(() => {
    const backendCategoryConfig = schema.category_config;
    return Array.from(categoryMap.keys()).sort((a, b) => {
      const orderA = getCategoryOrder(a, backendCategoryConfig);
      const orderB = getCategoryOrder(b, backendCategoryConfig);
      return orderA - orderB;
    });
  }, [categoryMap, schema.category_config]);

  // Also check for categories that should have special controls even if no fields
  const allCategories = useMemo(() => {
    const categorySet = new Set(sortedCategories);
    // Add special control categories if schema supports them
    if (schema.supports_vace) {
      categorySet.add("vace");
    }
    if (schema.supports_lora) {
      categorySet.add("lora");
    }
    if (schema.supports_vace) {
      categorySet.add("preprocessor");
    }
    const backendCategoryConfig = schema.category_config;
    return Array.from(categorySet).sort((a, b) => {
      const orderA = getCategoryOrder(a, backendCategoryConfig);
      const orderB = getCategoryOrder(b, backendCategoryConfig);
      return orderA - orderB;
    });
  }, [sortedCategories, schema]);

  return (
    <div className="space-y-4">
      {allCategories.map(categoryName => {
        const fields = categoryMap.get(categoryName) || [];
        const hasFields = fields.length > 0;
        const specialControl = (
          <SpecialControlRenderer
            categoryName={categoryName}
            schema={schema}
            fieldValues={fieldValues}
            inputMode={inputMode}
            disabled={disabled}
            props={{ ...specialControlProps, onFieldChange }}
          />
        );

        // If no fields and no special control, skip this category
        if (!hasFields && !specialControl) {
          return null;
        }

        // If we have fields, render them in a section (with special control if present)
        if (hasFields) {
          return (
            <DynamicSection
              key={categoryName}
              categoryName={categoryName}
              fields={fields}
              fieldValues={fieldValues}
              inputMode={inputMode}
              onFieldChange={onFieldChange}
              disabled={disabled}
              specialControl={specialControl}
              schema={schema}
            />
          );
        }

        // If only special control, render it without card wrapper
        if (specialControl) {
          return (
            <div key={categoryName} className="space-y-4">
              {specialControl}
            </div>
          );
        }

        return null;
      })}
    </div>
  );
}
