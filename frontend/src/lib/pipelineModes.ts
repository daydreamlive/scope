import { PIPELINES } from "../data/pipelines";
import { getCachedPipelineSchema } from "./utils";
import { GENERATION_MODE, type GenerationMode } from "../constants/modes";
import type { PipelineCategory, PipelineId } from "../types";

export interface PipelineModeCapabilities {
  id: PipelineId;
  category: PipelineCategory;
  nativeMode: GenerationMode;
  supportsVideo: boolean;
  supportsText: boolean;
  /**
   * Whether this pipeline requires an actual video stream when running in
   * video mode. If false, the pipeline can operate without input video even
   * when generationMode is set to video.
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
  const nativeMode: GenerationMode =
    cachedSchema?.native_mode ??
    info?.nativeGenerationMode ??
    (category === "video-input" ? GENERATION_MODE.VIDEO : GENERATION_MODE.TEXT);

  // Derive support flags from schema's supported_modes
  const supportsVideo =
    cachedSchema?.supported_modes?.includes(GENERATION_MODE.VIDEO) ??
    category === "video-input";
  const supportsText =
    cachedSchema?.supported_modes?.includes(GENERATION_MODE.TEXT) ?? true;

  const textConfig = cachedSchema?.mode_configs?.text;
  const videoConfig = cachedSchema?.mode_configs?.video;

  // Use backend-computed capabilities if available, otherwise derive from configs
  // This eliminates frontend-backend logic duplication
  const capabilities = cachedSchema?.capabilities;
  const requiresVideoInVideoMode =
    capabilities?.requiresVideoInVideoMode ??
    (category === "video-input" && supportsVideo);
  const showNoiseControlsInText =
    capabilities?.showNoiseControlsInText ??
    (textConfig?.noise_scale != null || textConfig?.noise_controller != null);
  const showNoiseControlsInVideo =
    capabilities?.showNoiseControlsInVideo ??
    (videoConfig?.noise_scale != null || videoConfig?.noise_controller != null);
  const hasGenerationModeControl =
    capabilities?.hasGenerationModeControl ?? (supportsVideo && supportsText);
  const hasNoiseControls =
    capabilities?.hasNoiseControls ??
    (showNoiseControlsInText || showNoiseControlsInVideo);
  const hasCacheManagement =
    capabilities?.hasCacheManagement ??
    (textConfig?.manage_cache != null || videoConfig?.manage_cache != null);

  return {
    id,
    category,
    nativeMode,
    supportsVideo,
    supportsText,
    requiresVideoInVideoMode,
    defaultResolutionByMode: {
      text: textConfig?.resolution?.default,
      video: videoConfig?.resolution?.default,
    },
    showNoiseControlsInText,
    showNoiseControlsInVideo,
    hasGenerationModeControl,
    hasNoiseControls,
    hasCacheManagement,
  };
}
