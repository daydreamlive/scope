import { PIPELINES } from "../data/pipelines";
import { getDefaultResolution, getCachedPipelineDefaults } from "./utils";
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

  // Get native mode from API defaults
  const cachedDefaults = getCachedPipelineDefaults(id);
  const nativeMode: "video" | "text" =
    cachedDefaults?.native_generation_mode ??
    info?.nativeGenerationMode ??
    (category === "video-input" ? "video" : "text");

  // For now we derive support flags from category and assume all pipelines
  // support text prompts. If we introduce pipelines that truly do not support
  // text in the future we can add explicit metadata for that.
  const supportsVideo = category === "video-input";
  const supportsText = true;

  const requiresVideoInVideoMode = category === "video-input" && supportsVideo;

  // Get resolution from defaults (may be undefined if not yet loaded)
  const textResolution = getDefaultResolution(id, "text");
  const videoResolution = getDefaultResolution(id, "video");

  let showNoiseControlsInText = false;
  let showNoiseControlsInVideo = false;
  let hasGenerationModeControl = false;
  let hasNoiseControls = false;
  let hasCacheManagement = false;

  if (id === "streamdiffusionv2") {
    // StreamDiffusionV2 uses noise controls only in video-to-video mode. In
    // text-to-video mode we use a fixed denoising schedule, matching LongLive
    // and Krea.
    showNoiseControlsInText = false;
    showNoiseControlsInVideo = true;
  } else if (id === "longlive" || id === "krea-realtime-video") {
    // LongLive and Krea only need noise controls in video-to-video workflows.
    showNoiseControlsInText = false;
    showNoiseControlsInVideo = true;
  }

  // All three main pipelines expose generation mode, noise controls and cache
  // management toggles.
  if (
    id === "streamdiffusionv2" ||
    id === "longlive" ||
    id === "krea-realtime-video"
  ) {
    hasGenerationModeControl = true;
    hasNoiseControls = true;
    hasCacheManagement = true;
  }

  return {
    id,
    category,
    nativeMode,
    supportsVideo,
    supportsText,
    requiresVideoInVideoMode,
    defaultResolutionByMode: {
      text: textResolution,
      video: videoResolution,
    },
    showNoiseControlsInText,
    showNoiseControlsInVideo,
    hasGenerationModeControl,
    hasNoiseControls,
    hasCacheManagement,
  };
}
