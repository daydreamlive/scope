import type { PipelineId, SettingsState } from "../types";
import { getModeDefaults } from "../lib/utils";
import { getPipelineModeCapabilities } from "../lib/pipelineModes";

/**
 * Get the settings update object for switching to a new pipeline.
 * This consolidates the logic for applying pipeline-specific defaults
 * including resolution, denoising steps, cache management, and noise controls.
 */
export function getPipelineSettingsUpdate(
  pipelineId: PipelineId
): Partial<SettingsState> {
  const caps = getPipelineModeCapabilities(pipelineId);
  const nativeGenerationMode = caps.nativeMode;
  const nativeModeDefaults = getModeDefaults(pipelineId, nativeGenerationMode);

  return {
    pipelineId,
    denoisingSteps: nativeModeDefaults.denoising_steps,
    resolution: nativeModeDefaults.resolution,
    generationMode: nativeGenerationMode,
    manageCache: nativeModeDefaults.manage_cache,
    noiseScale: nativeModeDefaults.noise_scale ?? undefined,
    noiseController: nativeModeDefaults.noise_controller ?? undefined,
    seed: nativeModeDefaults.base_seed,
    kvCacheAttentionBias:
      nativeModeDefaults.kv_cache_attention_bias ?? undefined,
  };
}
