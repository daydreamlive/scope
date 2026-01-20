/**
  "sdxl-turbo": {
    name: "SDXL Turbo",
    docsUrl:
      "",
    about:
      "",
    modified: true,
    category: "video-input",
    defaultPrompt:
      "A dog in the grass looking around, photorealistic",
    requiresModels: false,
    defaultTemporalInterpolationMethod: "linear",
    defaultTemporalInterpolationSteps: 4,
    pipelineCompatibility: "cloud",
    cloudModelId: "stabilityai/sdxl-turbo",
  },
  "sd-turbo": {
    name: "SDTurbo",
    docsUrl:
      "",
    about:
      "",
    modified: true,
    category: "video-input",
    defaultPrompt:
      "A dog in the grass looking around, photorealistic",
    requiresModels: false,
    defaultTemporalInterpolationMethod: "linear",
    defaultTemporalInterpolationSteps: 4,
    pipelineCompatibility: "cloud",
    cloudModelId: "stabilityai/sd-turbo",
  },
*/
import type { InputMode } from "../types";

// Default prompts by mode - used across all pipelines for consistency
export const DEFAULT_PROMPTS: Record<InputMode, string> = {
  text: "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  video:
    "A 3D animated scene. A **panda** sitting in the grass, looking around.",
};

export function getDefaultPromptForMode(mode: InputMode): string {
  return DEFAULT_PROMPTS[mode];
}
