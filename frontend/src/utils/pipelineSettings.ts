import type { PipelineId, SettingsState } from "../types";
import { getModeConfig } from "../lib/utils";
import { getPipelineModeCapabilities } from "../lib/pipelineModes";

/**
 * Get the settings update object for switching to a new pipeline.
 * This consolidates the logic for applying pipeline-specific configuration
 * including resolution, denoising steps, cache management, and noise controls.
 *
 * NOTE: This function requires pipeline schema to be loaded. If called before
 * schema is available, it will throw an error. Consider letting useStreamState
 * handle applying schema automatically via its useEffect instead.
 */
export function getPipelineSettingsUpdate(
  pipelineId: PipelineId
): Partial<SettingsState> {
  const caps = getPipelineModeCapabilities(pipelineId);
  const nativeGenerationMode = caps.nativeMode;

  // This will throw if schema isn't loaded yet - that's intentional
  // to catch misuse. Callers should ensure schema is loaded or use
  // the useStreamState hook which handles this automatically.
  const nativeModeConfig = getModeConfig(pipelineId, nativeGenerationMode);

  return {
    pipelineId,
    denoisingSteps: nativeModeConfig.denoising_steps ?? undefined,
    resolution: nativeModeConfig.resolution,
    generationMode: nativeGenerationMode,
    manageCache: nativeModeConfig.manage_cache,
    noiseScale: nativeModeConfig.noise_scale ?? undefined,
    noiseController: nativeModeConfig.noise_controller ?? undefined,
    seed: nativeModeConfig.base_seed,
    kvCacheAttentionBias: nativeModeConfig.kv_cache_attention_bias ?? undefined,
  };
}
