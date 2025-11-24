import type { PipelineId, SettingsState } from "../types";
import { getModeDefaults } from "../lib/utils";
import { getPipelineModeCapabilities } from "../lib/pipelineModes";

/**
 * Get the settings update object for switching to a new pipeline.
 * This consolidates the logic for applying pipeline-specific defaults
 * including resolution, denoising steps, cache management, and noise controls.
 *
 * NOTE: This function requires pipeline defaults to be loaded. If called before
 * defaults are available, it will throw an error. Consider letting useStreamState
 * handle applying defaults automatically via its useEffect instead.
 */
export function getPipelineSettingsUpdate(
  pipelineId: PipelineId
): Partial<SettingsState> {
  const caps = getPipelineModeCapabilities(pipelineId);
  const nativeGenerationMode = caps.nativeMode;

  // This will throw if defaults aren't loaded yet - that's intentional
  // to catch misuse. Callers should ensure defaults are loaded or use
  // the useStreamState hook which handles this automatically.
  const nativeModeDefaults = getModeDefaults(pipelineId, nativeGenerationMode);

  return {
    pipelineId,
    denoisingSteps: nativeModeDefaults.denoising_steps ?? undefined,
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
