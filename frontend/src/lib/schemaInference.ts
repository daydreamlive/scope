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
  | "unknown";

/**
 * Resolved schema property with enum values extracted from $defs.
 */
export interface ResolvedSchemaProperty extends PipelineSchemaProperty {
  resolvedEnum?: string[];
}

/**
 * Infers the appropriate control type from a JSON Schema property.
 *
 * Mapping rules:
 * - `$ref` to enum definition → select dropdown
 * - `type: "boolean"` → toggle
 * - `type: "number"` or `"integer"` with `minimum` AND `maximum` → slider
 * - `type: "number"` or `"integer"` (no bounds or partial bounds) → number input
 * - `type: "string"` → text input
 * - Unknown → fallback (logs warning)
 */
export function inferControlType(
  property: ResolvedSchemaProperty
): ControlType {
  // Check for enum reference first (highest priority)
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
        const hasMin =
          nonNullType.minimum !== undefined || property.minimum !== undefined;
        const hasMax =
          nonNullType.maximum !== undefined || property.maximum !== undefined;
        if (hasMin && hasMax) return "slider";
        return "number";
      }
      if (nonNullType.type === "string") return "text";
    }
  }

  // Standard type checks
  if (property.type === "boolean") {
    return "toggle";
  }

  if (property.type === "number" || property.type === "integer") {
    // Slider requires both min and max bounds
    if (property.minimum !== undefined && property.maximum !== undefined) {
      return "slider";
    }
    return "number";
  }

  if (property.type === "string") {
    return "text";
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
 * Resolves a schema property, extracting enum values from $defs if needed.
 */
export function resolveSchemaProperty(
  property: PipelineSchemaProperty,
  defs?: Record<string, { enum?: unknown[] }>
): ResolvedSchemaProperty {
  const resolved: ResolvedSchemaProperty = { ...property };

  if (property.$ref) {
    resolved.resolvedEnum = resolveEnumFromRef(property.$ref, defs);
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
  "denoising_steps", // Handled by DenoisingStepsSlider
  "ref_images", // Handled by file picker
  "loras", // Handled by LoRAManager
  // Fields that are handled elsewhere in the UI
  "modes",
]);

/**
 * Checks if a parameter should be rendered dynamically.
 */
export function shouldRenderParameter(paramName: string): boolean {
  return !EXCLUDED_PARAMETERS.has(paramName);
}

/**
 * Gets display metadata for a parameter from the schema.
 */
export function getParameterDisplayInfo(
  paramName: string,
  property: PipelineSchemaProperty
): { label: string; tooltip?: string } {
  // Convert snake_case to Title Case for label
  const label =
    paramName
      .split("_")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ") + ":";

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
    const controlType = inferControlType(resolved);

    // Skip unknown types for now (they would need custom handling)
    if (controlType === "unknown") {
      console.warn(
        `[schemaInference] Unknown control type for parameter "${name}":`,
        property
      );
      continue;
    }

    const displayInfo = getParameterDisplayInfo(name, property);

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
