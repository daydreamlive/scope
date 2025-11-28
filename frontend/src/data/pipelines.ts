/**
 * Static UI metadata for pipelines.
 *
 * This file contains frontend-only metadata that doesn't belong in the backend schema.
 * Dynamic/mode-dependent defaults (resolution, noise, prompts, temporal interpolation)
 * come from the backend pipeline schema via the API.
 */

export type PipelineCategory = "video-input" | "no-video-input";

export interface PipelineInfo {
  about: string;
  docsUrl?: string;
  modified?: boolean;
  category: PipelineCategory;
  estimatedVram?: number; // GB
  requiresModels?: boolean; // Whether this pipeline requires models to be downloaded
  supportsLoRA?: boolean; // Whether this pipeline supports LoRA adapters
}

export const PIPELINES: Record<string, PipelineInfo> = {
  streamdiffusionv2: {
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/streamdiffusionv2/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from the creators of the original StreamDiffusion project. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support streaming.",
    modified: true,
    category: "video-input",
    estimatedVram: 20,
    requiresModels: true,
    supportsLoRA: true,
  },
  longlive: {
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/longlive/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.",
    modified: true,
    category: "video-input",
    estimatedVram: 20,
    requiresModels: true,
    supportsLoRA: true,
  },
  "krea-realtime-video": {
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/krea_realtime_video/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Krea. The model is trained using Self-Forcing on Wan2.1 14b.",
    modified: true,
    category: "video-input",
    estimatedVram: 32,
    requiresModels: true,
    supportsLoRA: true,
  },
  passthrough: {
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    category: "video-input",
    requiresModels: false,
  },
};

export function pipelineSupportsLoRA(pipelineId: string): boolean {
  return PIPELINES[pipelineId]?.supportsLoRA === true;
}
