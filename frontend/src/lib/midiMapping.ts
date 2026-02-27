import type { PipelineConfigSchema, PipelineSchemaProperty } from "./api";
import type { MIDIMapping } from "../types/midi";

export interface MappableParameter {
  key: string;
  type: "number" | "boolean" | "enum" | "string";
  min?: number;
  max?: number;
  default?: unknown;
  enumValues?: string[];
  description?: string;
  isRuntimeParam?: boolean;
}

export function getMappableParameters(
  configSchema?: PipelineConfigSchema
): MappableParameter[] {
  if (!configSchema?.properties) return [];

  const mappable: MappableParameter[] = [];

  for (const [key, prop] of Object.entries(configSchema.properties)) {
    const schemaProp = prop as PipelineSchemaProperty;

    if (schemaProp.ui?.component && schemaProp.ui.component !== "image") {
      if (
        !["vace", "lora", "resolution", "cache", "denoising_steps", "noise", "quantization"].includes(
          schemaProp.ui.component
        )
      ) {
        continue;
      }
    }

    let paramType: MappableParameter["type"] = "string";
    let min: number | undefined;
    let max: number | undefined;
    let enumValues: string[] | undefined;

    if (schemaProp.enum) {
      paramType = "enum";
      enumValues = schemaProp.enum as string[];
    } else if (schemaProp.$ref) {
      paramType = "enum";
      if (configSchema.$defs) {
        const refName = schemaProp.$ref.split("/").pop();
        if (refName && configSchema.$defs[refName]) {
          const def = configSchema.$defs[refName] as { enum?: unknown[] };
          if (def.enum) enumValues = def.enum as string[];
        }
      }
    } else if (schemaProp.type === "boolean") {
      paramType = "boolean";
    } else if (schemaProp.type === "number" || schemaProp.type === "integer") {
      paramType = "number";
      min = schemaProp.minimum;
      max = schemaProp.maximum;
    } else if (schemaProp.type === "string") {
      paramType = "string";
    } else if (schemaProp.anyOf) {
      const nonNull = (schemaProp.anyOf as unknown[]).find(
        (t: unknown) =>
          typeof t === "object" &&
          t !== null &&
          (t as { type?: string }).type !== "null"
      ) as PipelineSchemaProperty | undefined;

      if (nonNull) {
        if (nonNull.type === "boolean") {
          paramType = "boolean";
        } else if (nonNull.type === "number" || nonNull.type === "integer") {
          paramType = "number";
          min = nonNull.minimum;
          max = nonNull.maximum;
        } else if (nonNull.enum) {
          paramType = "enum";
          enumValues = nonNull.enum as string[];
        }
      }
    }

    if (paramType === "string" && !enumValues) continue;

    mappable.push({
      key,
      type: paramType,
      min,
      max,
      default: schemaProp.default,
      enumValues,
      description: schemaProp.description,
      isRuntimeParam: schemaProp.ui?.is_load_param === false,
    });
  }

  return mappable;
}

export function getParameterRange(
  param: MappableParameter,
  defaultMin?: number,
  defaultMax?: number
): { min: number; max: number } {
  if (param.type !== "number") {
    return { min: defaultMin ?? 0, max: defaultMax ?? 1 };
  }
  return {
    min: param.min ?? defaultMin ?? 0,
    max: param.max ?? defaultMax ?? 1,
  };
}

export function validateMapping(
  mapping: MIDIMapping,
  mappableParams: MappableParameter[]
): { valid: boolean; error?: string } {
  if (mapping.type === "continuous" || mapping.type === "toggle") {
    if (!mapping.target.parameter) {
      return { valid: false, error: "Missing parameter target" };
    }
    const param = mappableParams.find((p) => p.key === mapping.target.parameter);
    if (!param) {
      return { valid: false, error: `Parameter not found: ${mapping.target.parameter}` };
    }
    if (mapping.type === "continuous" && param.type !== "number") {
      return { valid: false, error: `Parameter ${mapping.target.parameter} is not numeric` };
    }
    if (mapping.type === "toggle" && param.type !== "boolean") {
      return { valid: false, error: `Parameter ${mapping.target.parameter} is not boolean` };
    }
  }

  if (mapping.type === "enum_cycle") {
    if (!mapping.target.parameter || !mapping.target.values) {
      return { valid: false, error: "Missing parameter or values for enum_cycle" };
    }
    const param = mappableParams.find((p) => p.key === mapping.target.parameter);
    if (!param || param.type !== "enum") {
      return { valid: false, error: `Parameter ${mapping.target.parameter} is not an enum` };
    }
  }

  if (mapping.type === "trigger") {
    if (!mapping.target.action) {
      return { valid: false, error: "Missing action for trigger" };
    }
  }

  return { valid: true };
}
