import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";
import type { PipelineSchema, ModeConfig } from "./api";

type GenerationMode = "text" | "video";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Cache for pipeline schemas fetched from API
const pipelineSchemaCache = new Map<PipelineId, PipelineSchema>();

/**
 * Set cached schema for a pipeline. Called after fetching from API.
 */
export function setPipelineSchema(
  pipelineId: PipelineId,
  schema: PipelineSchema
): void {
  pipelineSchemaCache.set(pipelineId, schema);
}

/**
 * Get cached schema for a pipeline, or null if not cached.
 */
export function getCachedPipelineSchema(
  pipelineId: PipelineId
): PipelineSchema | null {
  return pipelineSchemaCache.get(pipelineId) || null;
}

/**
 * Get mode-specific config for a pipeline.
 * Throws if schema not yet loaded - caller should ensure schema is fetched first.
 */
export function getModeConfig(
  pipelineId: PipelineId,
  mode?: GenerationMode
): ModeConfig {
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

  return {
    ...modeConfig,
    denoising_steps: modeConfig.denoising_steps
      ? [...modeConfig.denoising_steps]
      : null,
    resolution: { ...modeConfig.resolution },
  };
}

export function getDefaultDenoisingSteps(
  pipelineId: PipelineId,
  mode?: GenerationMode
): number[] | undefined {
  const cached = getCachedPipelineSchema(pipelineId);
  if (!cached?.mode_configs) {
    return undefined;
  }

  const native_mode = cached.native_mode;
  const resolvedMode = mode ?? native_mode;
  const modeConfig =
    cached.mode_configs[resolvedMode] ?? cached.mode_configs[native_mode];

  if (!modeConfig || !modeConfig.denoising_steps) {
    return undefined;
  }

  return [...modeConfig.denoising_steps];
}

export function getDefaultResolution(
  pipelineId: PipelineId,
  mode?: GenerationMode
):
  | {
      height: number;
      width: number;
    }
  | undefined {
  const cached = getCachedPipelineSchema(pipelineId);
  if (!cached?.mode_configs) {
    return undefined;
  }

  const native_mode = cached.native_mode;
  const resolvedMode = mode ?? native_mode;
  const modeConfig =
    cached.mode_configs[resolvedMode] ?? cached.mode_configs[native_mode];

  if (!modeConfig) {
    return undefined;
  }

  return { ...modeConfig.resolution };
}
