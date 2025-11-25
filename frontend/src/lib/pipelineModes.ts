import { PIPELINES } from "../data/pipelines";
import { getCachedPipelineSchema } from "./utils";
import type { PipelineCategory, PipelineId } from "../types";

export interface PipelineModeCapabilities {
  id: PipelineId;
  category: PipelineCategory;
  nativeMode: "video" | "text";
  supportsVideo: boolean;
  supportsText: boolean;
  /**
   * Whether this pipeline requires an actual video stream when running in
   * video mode. If false, the pipeline can operate without input video even
   * when generationMode === "video".
   */
  requiresVideoInVideoMode: boolean;
  defaultResolutionByMode: {
    text?: {
      height: number;
      width: number;
    };
    video?: {
      height: number;
      width: number;
    };
  };
  showNoiseControlsInText: boolean;
  showNoiseControlsInVideo: boolean;
  /**
   * Whether this pipeline exposes an explicit generation_mode control that
   * allows switching between text-to-video and video-to-video behaviour.
   */
  hasGenerationModeControl: boolean;
  /**
   * Whether this pipeline supports user-adjustable noise controls. When false,
   * UI and initial parameter wiring should avoid sending noise_scale or
   * noise_controller.
   */
  hasNoiseControls: boolean;
  /**
   * Whether this pipeline exposes a cache management toggle (manage_cache).
   */
  hasCacheManagement: boolean;
}

export function getPipelineModeCapabilities(
  id: PipelineId
): PipelineModeCapabilities {
  const info = PIPELINES[id];
  const category: PipelineCategory = info?.category ?? "no-video-input";

  // Get schema from cache - this drives all capabilities
  const cachedSchema = getCachedPipelineSchema(id);
  const nativeMode: "video" | "text" =
    cachedSchema?.native_mode ??
    info?.nativeGenerationMode ??
    (category === "video-input" ? "video" : "text");

  // Derive support flags from schema's supported_modes
  const supportsVideo =
    cachedSchema?.supported_modes?.includes("video") ??
    category === "video-input";
  const supportsText = cachedSchema?.supported_modes?.includes("text") ?? true;

  const requiresVideoInVideoMode = category === "video-input" && supportsVideo;

  // Derive capabilities from schema's mode configs
  const textConfig = cachedSchema?.mode_configs?.text;
  const videoConfig = cachedSchema?.mode_configs?.video;

  // Noise controls are available if the mode config has non-null noise_scale or noise_controller
  const showNoiseControlsInText =
    textConfig?.noise_scale != null || textConfig?.noise_controller != null;
  const showNoiseControlsInVideo =
    videoConfig?.noise_scale != null || videoConfig?.noise_controller != null;

  // Has generation mode control if both modes are supported
  const hasGenerationModeControl = supportsVideo && supportsText;

  // Has noise controls if either mode supports them
  const hasNoiseControls = showNoiseControlsInText || showNoiseControlsInVideo;

  // Cache management is a property of mode configs
  const hasCacheManagement =
    textConfig?.manage_cache != null || videoConfig?.manage_cache != null;

  return {
    id,
    category,
    nativeMode,
    supportsVideo,
    supportsText,
    requiresVideoInVideoMode,
    defaultResolutionByMode: {
      text: textConfig?.resolution,
      video: videoConfig?.resolution,
    },
    showNoiseControlsInText,
    showNoiseControlsInVideo,
    hasGenerationModeControl,
    hasNoiseControls,
    hasCacheManagement,
  };
}
