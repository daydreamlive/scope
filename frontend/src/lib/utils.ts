import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";
import type { PipelineSchema } from "./api";

type GenerationMode = "text" | "video";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface CachedSchema {
  schema: PipelineSchema;
  timestamp: number;
  version: string;
}

// Cache for pipeline schemas fetched from API with expiration
const pipelineSchemaCache = new Map<PipelineId, CachedSchema>();

// Cache TTL: 5 minutes
const SCHEMA_CACHE_TTL = 5 * 60 * 1000;

/**
 * Set cached schema for a pipeline. Called after fetching from API.
 * Stores the schema with timestamp and version for cache invalidation.
 */
export function setPipelineSchema(
  pipelineId: PipelineId,
  schema: PipelineSchema
): void {
  pipelineSchemaCache.set(pipelineId, {
    schema,
    timestamp: Date.now(),
    version: schema.version,
  });
}

/**
 * Get cached schema for a pipeline, or null if not cached or expired.
 * Automatically removes expired entries from cache.
 */
export function getCachedPipelineSchema(
  pipelineId: PipelineId
): PipelineSchema | null {
  const cached = pipelineSchemaCache.get(pipelineId);
  if (!cached) {
    return null;
  }

  // Check if expired
  if (Date.now() - cached.timestamp > SCHEMA_CACHE_TTL) {
    pipelineSchemaCache.delete(pipelineId);
    return null;
  }

  return cached.schema;
}

/**
 * Invalidate cached schema for a pipeline.
 * Useful when pipeline is reloaded or updated.
 */
export function invalidatePipelineSchema(pipelineId: PipelineId): void {
  pipelineSchemaCache.delete(pipelineId);
}

/**
 * Clear all cached schemas.
 * Useful for development or when schemas might have changed.
 */
export function clearSchemaCache(): void {
  pipelineSchemaCache.clear();
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
  mode?: GenerationMode
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
  const extracted: ExtractedModeConfig = {
    resolution: modeConfig.resolution?.default ?? { height: 512, width: 512 },
    manage_cache: modeConfig.manage_cache?.default ?? true,
    base_seed: modeConfig.base_seed?.default ?? 42,
  };

  // Optional parameters
  if (modeConfig.denoising_steps) {
    extracted.denoising_steps = modeConfig.denoising_steps.default
      ? [...modeConfig.denoising_steps.default]
      : null;
  }
  if (modeConfig.noise_scale) {
    extracted.noise_scale = modeConfig.noise_scale.default ?? null;
  }
  if (modeConfig.noise_controller) {
    extracted.noise_controller = modeConfig.noise_controller.default ?? null;
  }
  if (modeConfig.kv_cache_attention_bias) {
    extracted.kv_cache_attention_bias =
      modeConfig.kv_cache_attention_bias.default ?? null;
  }
  if (modeConfig.input_size) {
    extracted.input_size = modeConfig.input_size.default ?? null;
  }
  if (modeConfig.vae_strategy) {
    extracted.vae_strategy = modeConfig.vae_strategy.default ?? null;
  }

  // Extract any extra parameters
  for (const key in modeConfig) {
    if (
      !Object.hasOwn(extracted, key) &&
      typeof modeConfig[key] === "object" &&
      modeConfig[key] !== null &&
      "default" in modeConfig[key]
    ) {
      extracted[key] = modeConfig[key].default as
        | number
        | boolean
        | string
        | number[]
        | { height: number; width: number }
        | null
        | undefined;
    }
  }

  return extracted;
}
