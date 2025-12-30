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
