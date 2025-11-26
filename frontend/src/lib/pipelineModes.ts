import { PIPELINES } from "../data/pipelines";
import { getCachedPipelineSchema } from "./utils";
import { INPUT_MODE, type InputMode } from "../constants/modes";
import type { PipelineId } from "../types";

export interface PipelineModeCapabilities {
  id: PipelineId;
  nativeMode: InputMode;
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
   * Whether this pipeline exposes an explicit input_mode control that
   * allows switching between text-to-video and video-to-video behaviour.
   */
  hasInputModeControl: boolean;
  /**
   * Whether this pipeline exposes a cache management toggle (manage_cache).
   */
  hasCacheManagement: boolean;
  /**
   * Whether this pipeline requires an actual video stream when running in
   * video mode. If false, the pipeline can operate without input video even
   * when inputMode is set to video.
   */
  requiresVideoInVideoMode: boolean;
}

/**
 * Whether this pipeline requires an actual video stream when running in
 * video mode. If false, the pipeline can operate without input video even
 * when inputMode is set to video.
 */
export function requiresVideoInVideoMode(
  caps: PipelineModeCapabilities
): boolean {
  return caps.requiresVideoInVideoMode;
}

/**
 * Whether this pipeline supports user-adjustable noise controls. When false,
 * UI and initial parameter wiring should avoid sending noise_scale or
 * noise_controller.
 */
export function hasNoiseControls(caps: PipelineModeCapabilities): boolean {
  return caps.showNoiseControlsInText || caps.showNoiseControlsInVideo;
}

/**
 * Resolve the effective input mode, falling back to the pipeline's native mode
 * when no explicit mode is set.
 */
export function getEffectiveMode(
  inputMode: InputMode | undefined,
  caps: PipelineModeCapabilities
): InputMode {
  return inputMode ?? caps.nativeMode;
}

/**
 * Whether the pipeline needs an actual video source for the given mode.
 * Returns true only when the pipeline requires video input AND we're in video mode.
 */
export function pipelineNeedsVideoSource(
  caps: PipelineModeCapabilities,
  effectiveMode: InputMode
): boolean {
  return requiresVideoInVideoMode(caps) && effectiveMode === INPUT_MODE.VIDEO;
}

export function getPipelineModeCapabilities(
  id: PipelineId
): PipelineModeCapabilities {
  const info = PIPELINES[id];
  const category = info?.category ?? "no-video-input";

  // Get schema from cache - this drives all capabilities
  const cachedSchema = getCachedPipelineSchema(id);
  const nativeMode: InputMode =
    cachedSchema?.native_mode ??
    info?.nativeInputMode ??
    (category === "video-input" ? INPUT_MODE.VIDEO : INPUT_MODE.TEXT);

  const textConfig = cachedSchema?.mode_configs?.text;
  const videoConfig = cachedSchema?.mode_configs?.video;

  // Use backend-computed capabilities directly
  const capabilities = cachedSchema?.capabilities;

  return {
    id,
    nativeMode,
    defaultResolutionByMode: {
      text: textConfig?.resolution?.default,
      video: videoConfig?.resolution?.default,
    },
    showNoiseControlsInText: capabilities?.showNoiseControlsInText ?? false,
    showNoiseControlsInVideo: capabilities?.showNoiseControlsInVideo ?? false,
    hasInputModeControl: capabilities?.hasInputModeControl ?? false,
    hasCacheManagement: capabilities?.hasCacheManagement ?? false,
    requiresVideoInVideoMode:
      capabilities?.requiresVideoInVideoMode ?? category === "video-input",
  };
}
