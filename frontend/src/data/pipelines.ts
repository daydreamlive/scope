export type PipelineCategory = "video-input" | "no-video-input";

export interface PipelineInfo {
  name: string;
  about: string;
  projectUrl?: string;
  modified?: boolean;
  category: PipelineCategory;
  defaultPrompt?: string;
}

export const PIPELINES: Record<string, PipelineInfo> = {
  longlive: {
    name: "LongLive",
    projectUrl: "https://nvlabs.github.io/LongLive/",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.",
    modified: true,
    category: "no-video-input",
    defaultPrompt:
      "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  },
  passthrough: {
    name: "Passthrough",
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    category: "video-input",
  },
  // vod: {
  //   name: "VOD",
  //   about:
  //     "A pipeline that returns a static video file without any processing that is useful for testing and debugging.",
  //   category: "no-video-input",
  // },
};
