/**
 * Schema inference utilities for dynamic control rendering.
 *
 * Maps JSON Schema types from Pydantic to UI control types.
 * This is the core logic that enables type-driven dynamic rendering.
 */

import type { PipelineSchemaProperty, PipelineConfigSchema } from "./api";

/**
 * Control types that can be rendered dynamically.
 */
export type ControlType =
  | "slider"
  | "number"
  | "toggle"
  | "select"
  | "text"
  | "textarea"
  | "denoisingSteps"
  | "unknown";

/**
 * Resolved schema property with enum values extracted from $defs.
 */
export interface ResolvedSchemaProperty extends PipelineSchemaProperty {
  resolvedEnum?: string[];
  /** UI metadata from json_schema_extra */
  uiMetadata?: import("../types").UIMetadata;
}

/**
 * Infers the appropriate control type from a JSON Schema property.
 *
 * Priority order:
 * 1. ui:widget override (if specified in UI metadata)
 * 2. Field name pattern matching (e.g., denoising_steps → denoisingSteps)
 * 3. Type inference based on JSON Schema
 *
 * Type inference rules:
 * - `$ref` to enum definition → select dropdown
 * - `type: "boolean"` → toggle
 * - `type: "number"` or `"integer"` with `minimum` AND `maximum` AND range ≤ 10 → slider
 * - `type: "number"` or `"integer"` (no bounds or large range) → number input (stepper)
 * - `type: "string"` with maxLength > 200 → textarea
 * - `type: "string"` → text input
 * - `type: "array"` with items.type: "integer" → denoisingSteps (if field name matches pattern)
 * - Unknown → fallback (logs warning)
 */
export function inferControlType(
  property: ResolvedSchemaProperty,
  fieldName?: string
): ControlType {
  // Priority 1: Check for ui:widget override
  if (property.uiMetadata?.["ui:widget"]) {
    const widget = property.uiMetadata["ui:widget"];
    // Map widget names to control types
    switch (widget) {
      case "slider":
      case "Slider":
        return "slider";
      case "number":
      case "stepper":
      case "Number":
        return "number";
      case "toggle":
      case "switch":
      case "Toggle":
        return "toggle";
      case "select":
      case "dropdown":
      case "Select":
        return "select";
      case "text":
      case "input":
      case "Text":
        return "text";
      case "textarea":
      case "Textarea":
        return "textarea";
      case "denoisingSteps":
      case "denoising_steps":
        return "denoisingSteps";
      default:
        // Custom widget types (like "loraManager") will be handled by the renderer
        return "unknown";
    }
  }

  // Priority 2: Field name pattern matching
  if (fieldName) {
    // Check for denoising_steps pattern
    if (
      fieldName === "denoising_steps" &&
      property.type === "array" &&
      property.items &&
      typeof property.items === "object" &&
      "type" in property.items &&
      property.items.type === "integer"
    ) {
      return "denoisingSteps";
    }
  }

  // Priority 3: Type inference

  // Check for enum reference first
  if (property.$ref || property.resolvedEnum) {
    return "select";
  }

  // Check for direct enum on property
  if (property.enum && Array.isArray(property.enum)) {
    return "select";
  }

  // Handle anyOf (often used for nullable types)
  if (property.anyOf && Array.isArray(property.anyOf)) {
    // Find the non-null type in anyOf
    const nonNullType = property.anyOf.find(
      (t: unknown) =>
        typeof t === "object" &&
        t !== null &&
        (t as { type?: string }).type !== "null"
    ) as { type?: string; minimum?: number; maximum?: number } | undefined;

    if (nonNullType) {
      if (nonNullType.type === "boolean") return "toggle";
      if (nonNullType.type === "number" || nonNullType.type === "integer") {
        // Check for bounds in the anyOf type or the parent property
        const min =
          nonNullType.minimum !== undefined
            ? nonNullType.minimum
            : property.minimum;
        const max =
          nonNullType.maximum !== undefined
            ? nonNullType.maximum
            : property.maximum;
        // Use slider if both bounds exist and range ≤ 10
        if (
          min !== undefined &&
          max !== undefined &&
          max - min <= 10
        ) {
          return "slider";
        }
        return "number";
      }
      if (nonNullType.type === "string") {
        // Check for textarea (long strings)
        const maxLength =
          "maxLength" in nonNullType && typeof nonNullType.maxLength === "number"
            ? nonNullType.maxLength
            : property.maxLength !== undefined && property.maxLength !== null
              ? property.maxLength
              : undefined;
        if (maxLength !== undefined && typeof maxLength === "number" && maxLength > 200) {
          return "textarea";
        }
        return "text";
      }
    }
  }

  // Standard type checks
  if (property.type === "boolean") {
    return "toggle";
  }

  if (property.type === "number" || property.type === "integer") {
    // Slider requires both min and max bounds AND range ≤ 10
    if (
      property.minimum !== undefined &&
      property.maximum !== undefined &&
      property.maximum - property.minimum <= 10
    ) {
      return "slider";
    }
    return "number";
  }

  if (property.type === "string") {
    // Check for textarea (long strings)
    const maxLength = property.maxLength;
    if (
      maxLength !== undefined &&
      typeof maxLength === "number" &&
      maxLength > 200
    ) {
      return "textarea";
    }
    return "text";
  }

  if (property.type === "array") {
    // Check if it's an array of integers (could be denoising steps)
    if (
      property.items &&
      typeof property.items === "object" &&
      "type" in property.items &&
      property.items.type === "integer"
    ) {
      // If field name wasn't checked earlier, check it now
      if (fieldName === "denoising_steps") {
        return "denoisingSteps";
      }
    }
  }

  // Unknown type - will use fallback renderer
  return "unknown";
}

/**
 * Resolves a $ref to its enum values from $defs.
 */
export function resolveEnumFromRef(
  ref: string,
  defs?: Record<string, { enum?: unknown[] }>
): string[] | undefined {
  if (!ref || !defs) return undefined;

  // Extract definition name from $ref (e.g., "#/$defs/VaeType" -> "VaeType")
  const defName = ref.split("/").pop();
  if (!defName) return undefined;

  const definition = defs[defName];
  if (definition && Array.isArray(definition.enum)) {
    return definition.enum as string[];
  }

  return undefined;
}

/**
 * Resolves a schema property, extracting enum values from $defs and UI metadata if needed.
 */
export function resolveSchemaProperty(
  property: PipelineSchemaProperty,
  defs?: Record<string, { enum?: unknown[] }>
): ResolvedSchemaProperty {
  const resolved: ResolvedSchemaProperty = { ...property };

  if (property.$ref) {
    resolved.resolvedEnum = resolveEnumFromRef(property.$ref, defs);
  }

  // Extract UI metadata from "x-ui" key (Pydantic json_schema_extra convention)
  // TypeScript doesn't recognize "x-ui" as a valid key, so we need to access it via bracket notation
  const xui = (property as Record<string, unknown>)["x-ui"];
  if (xui && typeof xui === "object") {
    resolved.uiMetadata = xui as import("../types").UIMetadata;
  }

  return resolved;
}

/**
 * Parameters that should be excluded from dynamic rendering.
 * These are either handled specially or are not user-facing.
 */
const EXCLUDED_PARAMETERS = new Set([
  // Internal/computed fields
  "pipeline_id",
  "pipeline_name",
  "pipeline_description",
  "pipeline_version",
  // Complex types that need custom handling
  "ref_images", // Handled by file picker
  "loras", // Handled by LoRAManager
  // Fields that are handled elsewhere in the UI
  "modes",
  // Note: denoising_steps is now handled via ui:widget="denoisingSteps" in UI metadata
]);

/**
 * Checks if a parameter should be rendered dynamically.
 */
export function shouldRenderParameter(paramName: string): boolean {
  return !EXCLUDED_PARAMETERS.has(paramName);
}

/**
 * Gets display metadata for a parameter from the schema.
 * Uses UI metadata if available, otherwise falls back to defaults.
 */
export function getParameterDisplayInfo(
  paramName: string,
  property: PipelineSchemaProperty | ResolvedSchemaProperty
): { label: string; tooltip?: string } {
  // Check for UI metadata override
  const uiMetadata =
    "uiMetadata" in property ? property.uiMetadata : undefined;

  // Use ui:label if provided, otherwise convert snake_case to Title Case
  const label =
    uiMetadata?.["ui:label"] ||
    paramName
      .split("_")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

  return {
    label,
    tooltip: property.description,
  };
}

/**
 * Extracts renderable parameters from a pipeline config schema.
 * Returns parameters in a format ready for dynamic rendering.
 */
export interface RenderableParameter {
  name: string;
  property: ResolvedSchemaProperty;
  controlType: ControlType;
  label: string;
  tooltip?: string;
}

export function extractRenderableParameters(
  schema: PipelineConfigSchema
): RenderableParameter[] {
  const parameters: RenderableParameter[] = [];

  if (!schema.properties) {
    return parameters;
  }

  for (const [name, property] of Object.entries(schema.properties)) {
    if (!shouldRenderParameter(name)) {
      continue;
    }

    const resolved = resolveSchemaProperty(property, schema.$defs);

    // Check if field is hidden via UI metadata
    if (resolved.uiMetadata?.["ui:hidden"] === true) {
      continue;
    }

    const controlType = inferControlType(resolved, name);

    // Skip unknown types for now (they would need custom handling)
    if (controlType === "unknown") {
      console.warn(
        `[schemaInference] Unknown control type for parameter "${name}":`,
        property
      );
      continue;
    }

    const displayInfo = getParameterDisplayInfo(name, resolved);

    parameters.push({
      name,
      property: resolved,
      controlType,
      label: displayInfo.label,
      tooltip: displayInfo.tooltip,
    });
  }

  return parameters;
}
