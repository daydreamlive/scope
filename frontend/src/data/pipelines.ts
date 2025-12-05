import type { InputMode } from "../types";

// Unified default prompts by mode (not per-pipeline)
// These are used across all pipelines for consistency
export const DEFAULT_PROMPTS: Record<InputMode, string> = {
  text: "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  video:
    "A 3D animated scene. A **panda** sitting in the grass, looking around.",
};

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

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;
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
    // Multi-mode support
    supportedModes: ["text", "video"],
    defaultMode: "video",
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
    // Multi-mode support
    supportedModes: ["text", "video"],
    defaultMode: "text",
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
    // Multi-mode support
    supportedModes: ["text", "video"],
    defaultMode: "text",
  },
  passthrough: {
    name: "Passthrough",
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    requiresModels: false,
    // Video-only pipeline
    supportedModes: ["video"],
    defaultMode: "video",
  },
  "reward-forcing": {
    name: "Reward-Forcing",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/reward_forcing/docs/usage.md",
    about:
      "A fast video generation pipeline using reward-based distillation. Generates high-quality videos in just 4 denoising steps. Based on Wan2.1 T2V 1.3B.",
    modified: true,
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
    supportsLoRA: true,
    // Multi-mode support
    supportedModes: ["text", "video"],
    defaultMode: "text",
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
