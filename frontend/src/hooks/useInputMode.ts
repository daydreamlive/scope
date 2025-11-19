import { useState, useCallback } from "react";
import type { PipelineCategory } from "../types";

export type InputMode = "video" | "camera" | "image" | "text";

interface UseInputModeProps {
  pipelineCategory: PipelineCategory;
  supportsImageInput?: boolean; // New prop to indicate if pipeline supports image input
}

export function useInputMode({
  pipelineCategory,
  supportsImageInput = false,
}: UseInputModeProps) {
  // Determine initial mode based on pipeline category
  const getInitialMode = (category: PipelineCategory): InputMode => {
    if (category === "video-input") {
      return "video";
    } else {
      return "text"; // For no-video-input pipelines, start with text mode
    }
  };

  const [mode, setMode] = useState<InputMode>(getInitialMode(pipelineCategory));

  const switchMode = useCallback((newMode: InputMode) => {
    setMode(newMode);
  }, []);

  // Get available modes based on pipeline category
  const getAvailableModes = (
    category: PipelineCategory,
    imageSupport: boolean
  ): InputMode[] => {
    if (category === "video-input") {
      // For video-input pipelines, include image if supported
      return imageSupport ? ["video", "camera", "image"] : ["video", "camera"];
    } else {
      return ["image", "text"];
    }
  };

  const availableModes = getAvailableModes(
    pipelineCategory,
    supportsImageInput
  );

  // Check if a mode is disabled
  const isModeDisabled = (modeToCheck: InputMode): boolean => {
    if (pipelineCategory === "no-video-input" && modeToCheck === "image") {
      return true; // Image mode is disabled for no-video-input pipelines (for now)
    }
    if (
      pipelineCategory === "video-input" &&
      modeToCheck === "image" &&
      !supportsImageInput
    ) {
      return true; // Image mode is disabled if pipeline doesn't support it
    }
    return false;
  };

  return {
    mode,
    switchMode,
    availableModes,
    isModeDisabled,
  };
}
