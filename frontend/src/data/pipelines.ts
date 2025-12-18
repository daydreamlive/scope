import type { InputMode } from "../types";

// Unified default prompts by mode (not per-pipeline)
// These are used across all pipelines for consistency
export const DEFAULT_PROMPTS: Record<InputMode, string> = {
  text: "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  video:
    "A 3D animated scene. A **panda** sitting in the grass, looking around.",
};

// UI capability flags - controls which UI elements are shown for each pipeline
export interface PipelineUICapabilities {
  showTimeline?: boolean; // Show prompt timeline (default: true)
  showPromptInput?: boolean; // Show prompt input controls (default: true)
  showResolutionControl?: boolean; // Show resolution width/height controls
  showSeedControl?: boolean; // Show seed control
  showDenoisingSteps?: boolean; // Show denoising steps slider
  showCacheManagement?: boolean; // Show cache management toggle and reset
  showQuantization?: boolean; // Show quantization selector
  showKvCacheAttentionBias?: boolean; // Show KV cache attention bias slider (krea-realtime-video)
  showNoiseControls?: boolean; // Show noise scale and controller (video mode)
  canChangeReferenceWhileStreaming?: boolean; // Allow changing reference image during stream
}

export interface PipelineInfo {
  name: string;
  about: string;
  docsUrl?: string;
  modified?: boolean;
  estimatedVram?: number; // GB
  requiresModels?: boolean; // Whether this pipeline requires models to be downloaded
  defaultTemporalInterpolationMethod?: "linear" | "slerp"; // Default method for temporal interpolation
  defaultTemporalInterpolationSteps?: number; // Default number of steps for temporal interpolation
  supportsLoRA?: boolean; // Whether this pipeline supports LoRA adapters
  requiresReferenceImage?: boolean; // Whether this pipeline requires a reference image (e.g. PersonaLive)
  referenceImageDescription?: string; // Description of what the reference image is for

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;

  // UI capabilities - controls which settings/controls are shown
  ui?: PipelineUICapabilities;
}

export const PIPELINES: Record<string, PipelineInfo> = {
  streamdiffusionv2: {
    name: "StreamDiffusionV2",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/streamdiffusionv2/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from the creators of the original StreamDiffusion project. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support streaming.",
    modified: true,
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
    supportsLoRA: true,
    supportedModes: ["text", "video"],
    defaultMode: "video",
    ui: {
      showTimeline: true,
      showPromptInput: true,
      showResolutionControl: true,
      showSeedControl: true,
      showDenoisingSteps: true,
      showCacheManagement: true,
      showQuantization: true,
      showNoiseControls: true,
    },
  },
  longlive: {
    name: "LongLive",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/longlive/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.",
    modified: true,
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
    supportsLoRA: true,
    supportedModes: ["text", "video"],
    defaultMode: "text",
    ui: {
      showTimeline: true,
      showPromptInput: true,
      showResolutionControl: true,
      showSeedControl: true,
      showDenoisingSteps: true,
      showCacheManagement: true,
      showQuantization: true,
      showNoiseControls: true,
    },
  },
  "krea-realtime-video": {
    name: "Krea Realtime Video",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/krea_realtime_video/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Krea. The model is trained using Self-Forcing on Wan2.1 14b.",
    modified: true,
    estimatedVram: 32,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "linear",
    defaultTemporalInterpolationSteps: 4,
    supportsLoRA: true,
    supportedModes: ["text", "video"],
    defaultMode: "text",
    ui: {
      showTimeline: true,
      showPromptInput: true,
      showResolutionControl: true,
      showSeedControl: true,
      showDenoisingSteps: true,
      showCacheManagement: true,
      showQuantization: true,
      showKvCacheAttentionBias: true,
      showNoiseControls: true,
    },
  },
  "reward-forcing": {
    name: "RewardForcing",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/reward_forcing/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from ZJU, Ant Group, SIAS-ZJU, HUST and SJTU. The model is trained with Rewarded Distribution Matching Distillation using Wan2.1 1.3b as the base model.",
    modified: true,
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
    supportsLoRA: true,
    supportedModes: ["text", "video"],
    defaultMode: "text",
    ui: {
      showTimeline: true,
      showPromptInput: true,
      showResolutionControl: true,
      showSeedControl: true,
      showDenoisingSteps: true,
      showCacheManagement: true,
      showQuantization: true,
      showNoiseControls: true,
    },
  },
  passthrough: {
    name: "Passthrough",
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    requiresModels: false,
    supportedModes: ["video"],
    defaultMode: "video",
    ui: {
      showTimeline: false,
      showPromptInput: false,
      showResolutionControl: false,
      showSeedControl: false,
      showDenoisingSteps: false,
      showCacheManagement: false,
      showQuantization: false,
      showNoiseControls: false,
    },
  },
  personalive: {
    name: "PersonaLive",
    docsUrl: "https://github.com/GVCLab/PersonaLive",
    about:
      "Real-time portrait animation pipeline from GVCLab. Animates a reference portrait image using driving video frames to transfer expressions and head movements.",
    estimatedVram: 12,
    requiresModels: true,
    supportedModes: ["video"],
    defaultMode: "video",
    requiresReferenceImage: true,
    referenceImageDescription:
      "Portrait image to animate. Expressions and head movements from the driving video will be transferred to this image.",
    ui: {
      showTimeline: false,
      showPromptInput: false,
      showResolutionControl: true,
      showSeedControl: true,
      showDenoisingSteps: false,
      showCacheManagement: false,
      showQuantization: false,
      showNoiseControls: false,
      canChangeReferenceWhileStreaming: false,
    },
  },
};

export function pipelineSupportsLoRA(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.supportsLoRA === true;
}

export function pipelineSupportsMode(
  pipelineId: string,
  mode: InputMode
): boolean {
  return PIPELINES[pipelineId]?.supportedModes?.includes(mode) ?? false;
}

export function pipelineIsMultiMode(pipelineId: string): boolean {
  const modes = PIPELINES[pipelineId]?.supportedModes ?? [];
  return modes.length > 1;
}

export function getPipelineDefaultMode(pipelineId: string): InputMode {
  return PIPELINES[pipelineId]?.defaultMode ?? "text";
}

export function getDefaultPromptForMode(mode: InputMode): string {
  return DEFAULT_PROMPTS[mode];
}

export function pipelineRequiresReferenceImage(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.requiresReferenceImage === true;
}

export function getPipelineReferenceImageDescription(
  pipelineId: string
): string | undefined {
  return PIPELINES[pipelineId]?.referenceImageDescription;
}

// UI capability helper functions

export function pipelineShowsTimeline(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showTimeline !== false;
}

export function pipelineShowsPromptInput(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showPromptInput !== false;
}

export function pipelineShowsResolutionControl(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showResolutionControl === true;
}

export function pipelineShowsSeedControl(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showSeedControl === true;
}

export function pipelineShowsDenoisingSteps(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showDenoisingSteps === true;
}

export function pipelineShowsCacheManagement(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showCacheManagement === true;
}

export function pipelineShowsQuantization(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showQuantization === true;
}

export function pipelineShowsKvCacheAttentionBias(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showKvCacheAttentionBias === true;
}

export function pipelineShowsNoiseControls(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.ui?.showNoiseControls === true;
}

export function pipelineCanChangeReferenceWhileStreaming(
  pipelineId: string
): boolean {
  return PIPELINES[pipelineId]?.ui?.canChangeReferenceWhileStreaming !== false;
}
