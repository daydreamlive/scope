import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";
import type { PipelineSchema } from "./api";

type InputMode = "text" | "video";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface CachedSchema {
  schema: PipelineSchema;
}

// Cache for pipeline schemas fetched from API
const pipelineSchemaCache = new Map<PipelineId, CachedSchema>();

/**
 * Set cached schema for a pipeline. Called after fetching from API.
 */
export function setPipelineSchema(
  pipelineId: PipelineId,
  schema: PipelineSchema
): void {
  pipelineSchemaCache.set(pipelineId, { schema });
}

/**
 * Get cached schema for a pipeline, or null if not cached.
 */
export function getCachedPipelineSchema(
  pipelineId: PipelineId
): PipelineSchema | null {
  const cached = pipelineSchemaCache.get(pipelineId);
  return cached?.schema ?? null;
}

/**
 * Extracted mode config with plain values (defaults extracted from JSON Schema)
 */
export interface ExtractedModeConfig {
  denoising_steps?: number[] | null;
  resolution: { height: number; width: number };
  manage_cache: boolean;
  base_seed: number;
  noise_scale?: number | null;
  noise_controller?: boolean | null;
  kv_cache_attention_bias?: number | null;
  input_size?: number | null;
  vae_strategy?: string | null;
  default_prompt?: string | null;
  default_temporal_interpolation_method?: "linear" | "slerp" | null;
  default_temporal_interpolation_steps?: number | null;
  [key: string]:
    | number
    | boolean
    | string
    | number[]
    | { height: number; width: number }
    | null
    | undefined;
}

/**
 * Get mode-specific config for a pipeline with extracted default values.
 * Throws if schema not yet loaded - caller should ensure schema is fetched first.
 */
export function getModeConfig(
  pipelineId: PipelineId,
  mode?: InputMode
): ExtractedModeConfig {
  const cached = getCachedPipelineSchema(pipelineId);
  if (!cached?.mode_configs) {
    throw new Error(
      `getModeConfig: Schema not loaded for ${pipelineId}. App should fetch schema before rendering.`
    );
  }

  const native_mode = cached.native_mode;
  const resolvedMode = mode ?? native_mode;
  const modeConfig =
    cached.mode_configs[resolvedMode] ?? cached.mode_configs[native_mode];

  if (!modeConfig) {
    throw new Error(
      `getModeConfig: No config for mode ${resolvedMode} in ${pipelineId}`
    );
  }

  // Extract default values from JSON Schema objects
  const extracted: ExtractedModeConfig = {} as ExtractedModeConfig;

  // Iterate over all fields in modeConfig
  for (const key in modeConfig) {
    const param = modeConfig[key];
    if (typeof param === "object" && param !== null && "default" in param) {
      // Special handling for denoising_steps: copy array to avoid mutation
      if (key === "denoising_steps" && Array.isArray(param.default)) {
        extracted[key] = [...param.default] as number[];
      } else {
        // Extract default value from schema
        extracted[key] = param.default as
          | number
          | boolean
          | string
          | number[]
          | { height: number; width: number }
          | null
          | undefined;
      }
    }
  }

  // Ensure required fields have defaults (in case they weren't in modeConfig or had null defaults)
  if (!extracted.resolution) {
    extracted.resolution = { height: 512, width: 512 };
  }
  if (extracted.manage_cache === undefined) {
    extracted.manage_cache = true;
  }
  if (extracted.base_seed === undefined) {
    extracted.base_seed = 42;
  }

  return extracted;
}
