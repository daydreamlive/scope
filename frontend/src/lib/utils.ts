import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";
import type { PipelineDefaults, PipelineModeDefaults } from "./api";

type GenerationMode = "text" | "video";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Cache for pipeline defaults fetched from API
const pipelineDefaultsCache = new Map<PipelineId, PipelineDefaults>();

/**
 * Set cached defaults for a pipeline. Called after fetching from API.
 */
export function setPipelineDefaults(
  pipelineId: PipelineId,
  defaults: PipelineDefaults
): void {
  pipelineDefaultsCache.set(pipelineId, defaults);
}

/**
 * Get cached defaults for a pipeline, or null if not cached.
 */
export function getCachedPipelineDefaults(
  pipelineId: PipelineId
): PipelineDefaults | null {
  return pipelineDefaultsCache.get(pipelineId) || null;
}

/**
 * Get mode-specific defaults for a pipeline.
 * Throws if defaults not yet loaded - caller should ensure defaults are fetched first.
 */
export function getModeDefaults(
  pipelineId: PipelineId,
  mode?: GenerationMode
): PipelineModeDefaults {
  const cached = getCachedPipelineDefaults(pipelineId);
  if (!cached?.modes) {
    throw new Error(
      `getModeDefaults: Defaults not loaded for ${pipelineId}. App should fetch defaults before rendering.`
    );
  }

  const native_mode = cached.native_generation_mode;
  const resolvedMode = mode ?? native_mode;
  const modeDefaults = cached.modes[resolvedMode] ?? cached.modes[native_mode];

  if (!modeDefaults) {
    throw new Error(
      `getModeDefaults: No defaults for mode ${resolvedMode} in ${pipelineId}`
    );
  }

  return {
    ...modeDefaults,
    denoising_steps: modeDefaults.denoising_steps
      ? [...modeDefaults.denoising_steps]
      : null,
    resolution: { ...modeDefaults.resolution },
  };
}

export function getDefaultDenoisingSteps(
  pipelineId: PipelineId,
  mode?: GenerationMode
): number[] | undefined {
  const cached = getCachedPipelineDefaults(pipelineId);
  if (!cached?.modes) {
    return undefined;
  }

  const native_mode = cached.native_generation_mode;
  const resolvedMode = mode ?? native_mode;
  const modeDefaults = cached.modes[resolvedMode] ?? cached.modes[native_mode];

  if (!modeDefaults || !modeDefaults.denoising_steps) {
    return undefined;
  }

  return [...modeDefaults.denoising_steps];
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
  const cached = getCachedPipelineDefaults(pipelineId);
  if (!cached?.modes) {
    return undefined;
  }

  const native_mode = cached.native_generation_mode;
  const resolvedMode = mode ?? native_mode;
  const modeDefaults = cached.modes[resolvedMode] ?? cached.modes[native_mode];

  if (!modeDefaults) {
    return undefined;
  }

  return { ...modeDefaults.resolution };
}
