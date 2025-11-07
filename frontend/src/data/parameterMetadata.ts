/**
 * Parameter metadata including labels and tooltip descriptions
 * for the SettingsPanel and related components.
 *
 * This centralized configuration makes it easy to maintain
 * parameter descriptions across the application.
 */

export interface ParameterMetadata {
  label: string;
  tooltip: string;
}

export const PARAMETER_METADATA: Record<string, ParameterMetadata> = {
  height: {
    label: "Height:",
    tooltip:
      "Output video height in pixels. Higher values produce more detailed vertical resolution but reduces speed.",
  },
  width: {
    label: "Width:",
    tooltip:
      "Output video width in pixels. Higher values produce more detailed horizontal resolution but reduces speed.",
  },
  seed: {
    label: "Seed:",
    tooltip:
      "Random seed for reproducible generation. Using the same seed with the same settings will produce similar results.",
  },
  manageCache: {
    label: "Manage Cache:",
    tooltip:
      "Enables pipeline to automatically manage the cache which influences newly generated frames. Disable for manual control via Reset Cache.",
  },
  resetCache: {
    label: "Reset Cache:",
    tooltip:
      "Clears previous frames from cache allowing new frames to be generated with fresh history. Only available when Manage Cache is disabled.",
  },
  denoisingSteps: {
    label: "Denoising Step List",
    tooltip:
      "List of denoising timesteps used in diffusion. Values must be in descending order. Lower values mean less noise to remove. More steps can improve quality but reduce speed.",
  },
  noiseController: {
    label: "Noise Controller:",
    tooltip:
      "Enables automatic noise scale adjustment based on detected motion. Disable for manual control via Noise Scale.",
  },
  noiseScale: {
    label: "Noise Scale:",
    tooltip:
      "Controls the amount of noise added during generation. Higher values add more variation and creativity and lower values produce more stable results.",
  },
  quantization: {
    label: "Quantization:",
    tooltip:
      "Quantization method for the diffusion model. fp8_e4m3fn reduces memory usage and improves performance, but might affect quality. None uses full precision and uses more meory, but does not affect quality.",
  },
  kvCacheAttentionBias: {
    label: "Cache Bias:",
    tooltip:
      "Controls how much to rely on past frames in the cache during generation. A lower value can help mitigate error accumulation and prevent repetitive motion. Uses log scale: 1.0 = full reliance on past frames, smaller values = less reliance on past frames. Typical values: 0.3-0.7 for moderate effect, 0.1-0.2 for strong effect.",
  },
};
