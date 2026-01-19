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
      "Quantization method for the diffusion model. fp8_e4m3fn (Dynamic) reduces memory usage, but might affect performance and quality. None uses full precision and uses more memory, but does not affect performance and quality.",
  },
  kvCacheAttentionBias: {
    label: "Cache Bias:",
    tooltip:
      "Controls how much to rely on past frames in the cache during generation. A lower value can help mitigate error accumulation and prevent repetitive motion. Uses log scale: 1.0 = full reliance on past frames, smaller values = less reliance on past frames. Typical values: 0.3-0.7 for moderate effect, 0.1-0.2 for strong effect.",
  },
  loraMergeStrategy: {
    label: "LoRA Strategy:",
    tooltip:
      "LoRA merge strategy affects performance and update capabilities. Permanent Merge: Maximum performance, no runtime updates. Runtime PEFT: Lower performance, instant runtime updates.",
  },
  loraScale: {
    label: "Scale:",
    tooltip:
      "Adjust LoRA strength. Updates automatically when you release the slider or use +/- buttons. Typical values: 0.0 = no effect, 1.0 = full strength. Full range -10.0 to 10.0 available depending on LoRA specifications.",
  },
  loraScaleDisabledDuringStream: {
    label: "Scale:",
    tooltip:
      "Runtime adjustment is disabled with Permanent Merge strategy. LoRA scales are fixed at load time. Typical values: 0.0 = no effect, 1.0 = full strength. Full range -10.0 to 10.0 available depending on LoRA specifications.",
  },
  spoutSender: {
    label: "Spout Sender:",
    tooltip:
      "The configuration of the sender that will send video to Spout-compatible apps like TouchDesigner, Resolume, OBS.",
  },
  vaeType: {
    label: "VAE:",
    tooltip:
      "VAE type to use for encoding/decoding. 'wan' is the full VAE with best quality. 'lightvae' is 75% pruned for faster performance but lower quality. 'tae' is a tiny autoencoder for fast preview quality. 'lighttae' is LightTAE with WanVAE normalization for faster performance with consistent latent space.",
  },
  preprocessor: {
    label: "Preprocessor:",
    tooltip: "Select a preprocessor to apply before the main pipeline.",
  },
  postprocessor: {
    label: "Postprocessor:",
    tooltip: "Select a postprocessor to apply after the main pipeline.",
  },
};
