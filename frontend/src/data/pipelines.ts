export type PipelineCategory = "video-input" | "no-video-input";

export interface PipelineInfo {
  name: string;
  about: string;
  docsUrl?: string;
  modified?: boolean;
  category: PipelineCategory;
  defaultPrompt?: string;
  estimatedVram?: number; // GB
  requiresModels?: boolean; // Whether this pipeline requires models to be downloaded
  defaultTemporalInterpolationMethod?: "linear" | "slerp"; // Default method for temporal interpolation
  defaultTemporalInterpolationSteps?: number; // Default number of steps for temporal interpolation
}

export const PIPELINES: Record<string, PipelineInfo> = {
  streamdiffusionv2: {
    name: "StreamDiffusionV2",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/streamdiffusionv2/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from the creators of the original StreamDiffusion project. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support streaming.",
    modified: true,
    category: "video-input",
    defaultPrompt: "A dog in the grass looking around, photorealistic",
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
  },
  longlive: {
    name: "LongLive",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/longlive/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.",
    modified: true,
    category: "no-video-input",
    defaultPrompt:
      "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
    estimatedVram: 20,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "slerp",
    defaultTemporalInterpolationSteps: 0,
  },
  "krea-realtime-video": {
    name: "Krea Realtime Video",
    docsUrl:
      "https://github.com/daydreamlive/scope/blob/main/pipelines/krea_realtime_video/docs/usage.md",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Krea. The model is trained using Self-Forcing on Wan2.1 14b.",
    modified: true,
    category: "no-video-input",
    defaultPrompt:
      "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
    estimatedVram: 32,
    requiresModels: true,
    defaultTemporalInterpolationMethod: "linear",
    defaultTemporalInterpolationSteps: 4,
  },
  passthrough: {
    name: "Passthrough",
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    category: "video-input",
    requiresModels: false,
  },
  // vod: {
  //   name: "VOD",
  //   about:
  //     "A pipeline that returns a static video file without any processing that is useful for testing and debugging.",
  //   category: "no-video-input",
  // },
};
