/**
 * Utilities for schema-driven settings UI.
 * - ui.category === "configuration" (or undefined) => Settings panel
 * - ui.category === "input" => Input & Controls panel, below app-defined sections (Prompts, etc.)
 * - If category is missing, it is treated as "configuration".
 */

export interface SchemaFieldUI {
  category?: string;
  order?: number;
  component?: string;
  modes?: ("text" | "video")[];
  /** If true, field is a load param (disabled when streaming); if false, runtime param (editable when streaming). Omit = treated as load param. */
  is_load_param?: boolean;
  /** Short label for the UI. When set, used instead of description for the field label. */
  label?: string;
  /** For component="custom": URL to load the web component script from */
  script_url?: string;
  /** For component="custom": Custom element tag name (e.g., "image-filter-layers") */
  element_name?: string;
}

export interface SchemaProperty {
  type?: string;
  default?: unknown;
  description?: string;
  minimum?: number;
  maximum?: number;
  enum?: unknown[];
  $ref?: string;
  ui?: SchemaFieldUI;
  [k: string]: unknown;
}

/** Accepts pipeline config schema from API; only requires properties[].ui for filtering. */
export type ConfigSchemaLike = {
  properties?: Record<string, SchemaProperty>;
  $defs?: Record<string, { enum?: unknown[] }>;
};

export interface ConfigurationField {
  key: string;
  prop: SchemaProperty;
  ui: SchemaFieldUI;
}

/** Primitive field types inferred from schema (used for widget selection). */
export type PrimitiveFieldType =
  | "text"
  | "number"
  | "slider"
  | "toggle"
  | "enum";

/** Parsed config field with resolved fieldType (primitive or complex component name). */
export interface ParsedFieldConfig extends ConfigurationField {
  fieldType: PrimitiveFieldType | ComplexComponentName;
}

/**
 * Infer primitive field type from a schema property.
 * Handles anyOf (nullable), $ref, enum, and direct type + min/max.
 */
export function inferPrimitiveFieldType(
  property: SchemaProperty
): PrimitiveFieldType | null {
  type Sub = {
    type?: string;
    enum?: unknown[];
    minimum?: number;
    maximum?: number;
  };

  function fromSub(sub: Sub): PrimitiveFieldType | null {
    if (!sub) return null;
    if (sub.enum) return "enum";
    if (sub.type === "string") return "text";
    if (sub.type === "integer" || sub.type === "number") {
      if (sub.minimum !== undefined && sub.maximum !== undefined) {
        return "slider";
      }
      return "number";
    }
    if (sub.type === "boolean") return "toggle";
    return null;
  }

  const anyOf = property.anyOf as unknown[] | undefined;
  if (anyOf?.length) {
    const nonNull = anyOf.find(
      (t: unknown) =>
        typeof t === "object" && t !== null && (t as Sub).type !== "null"
    ) as Sub | undefined;
    return nonNull ? fromSub(nonNull) : null;
  }

  if (property.$ref) return "enum";
  if (property.enum) return "enum";

  return fromSub(property as Sub);
}

/** Internal: resolve fieldType for a list of fields. */
function resolveFieldTypes(fields: ConfigurationField[]): ParsedFieldConfig[] {
  const result: ParsedFieldConfig[] = [];
  for (const { key, prop, ui } of fields) {
    let fieldType: PrimitiveFieldType | ComplexComponentName;
    if (
      ui.component &&
      COMPLEX_COMPONENTS.includes(ui.component as ComplexComponentName)
    ) {
      fieldType = ui.component as ComplexComponentName;
    } else {
      const inferred = inferPrimitiveFieldType(prop);
      if (!inferred) {
        console.warn(`Could not infer field type for ${key}, skipping`, prop);
        continue;
      }
      fieldType = inferred;
    }
    result.push({ key, prop, ui, fieldType });
  }
  return result;
}

/**
 * Parse configuration fields with resolved fieldType (component or inferred primitive).
 */
export function parseConfigurationFields(
  configSchema: ConfigSchemaLike | undefined,
  inputMode: "text" | "video" | undefined
): ParsedFieldConfig[] {
  return resolveFieldTypes(getConfigurationFields(configSchema, inputMode));
}

/** Effective category: missing means "configuration". */
function effectiveCategory(ui: SchemaFieldUI | undefined): string {
  return ui?.category ?? "configuration";
}

/**
 * Extract fields from pipeline config schema by category. Default category is "configuration".
 * Only properties with json_schema_extra (i.e. a "ui" key) are included; base schema fields
 * without explicit UI metadata are omitted. Filtered by input mode, sorted by ui.order.
 */
function getFieldsByCategory(
  configSchema: ConfigSchemaLike | undefined,
  inputMode: "text" | "video" | undefined,
  category: "configuration" | "input"
): ConfigurationField[] {
  const properties = configSchema?.properties ?? {};
  const fields: ConfigurationField[] = [];

  for (const [key, prop] of Object.entries(properties)) {
    const ui = (prop as SchemaProperty)?.ui;
    if (ui == null) continue; // only render fields that have json_schema_extra
    if (effectiveCategory(ui) !== category) continue;
    if (
      ui?.modes &&
      ui.modes.length > 0 &&
      inputMode &&
      !ui.modes.includes(inputMode)
    )
      continue;
    fields.push({ key, prop: prop as SchemaProperty, ui: ui ?? {} });
  }

  fields.sort((a, b) => {
    const oA = a.ui.order ?? 999;
    const oB = b.ui.order ?? 999;
    if (oA !== oB) return oA - oB;
    return a.key.localeCompare(b.key);
  });

  return fields;
}

/**
 * Configuration fields (category "configuration" or undefined) for the Settings panel.
 */
export function getConfigurationFields(
  configSchema: ConfigSchemaLike | undefined,
  inputMode: "text" | "video" | undefined
): ConfigurationField[] {
  return getFieldsByCategory(configSchema, inputMode, "configuration");
}

/**
 * Input fields (category "input") for the Input & Controls panel, shown below Prompts.
 */
export function getInputFields(
  configSchema: ConfigSchemaLike | undefined,
  inputMode: "text" | "video" | undefined
): ConfigurationField[] {
  return getFieldsByCategory(configSchema, inputMode, "input");
}

/**
 * Parse input fields (category "input") with resolved fieldType.
 */
export function parseInputFields(
  configSchema: ConfigSchemaLike | undefined,
  inputMode: "text" | "video" | undefined
): ParsedFieldConfig[] {
  return resolveFieldTypes(getInputFields(configSchema, inputMode));
}

/** Complex component names that render a single block (render once per component). "image" renders one picker per field. */
export const COMPLEX_COMPONENTS = [
  "vace",
  "lora",
  "resolution",
  "cache",
  "denoising_steps",
  "noise",
  "quantization",
  "image",
  "custom",
] as const;

export type ComplexComponentName = (typeof COMPLEX_COMPONENTS)[number];

/**
 * Check if a component is a custom web component (has script_url and element_name).
 */
export function isCustomWebComponent(ui?: SchemaFieldUI): boolean {
  return ui?.component === "custom" && !!ui?.script_url && !!ui?.element_name;
}
