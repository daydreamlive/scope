import { PIPELINES } from "../data/pipelines";
import { getDefaultResolution } from "./utils";
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
}

export function getPipelineModeCapabilities(
  id: PipelineId
): PipelineModeCapabilities {
  const info = PIPELINES[id];
  const category: PipelineCategory = info?.category ?? "no-video-input";

  const nativeMode: "video" | "text" =
    info?.nativeGenerationMode ??
    (category === "video-input" ? "video" : "text");

  // For now we derive support flags from category and assume all pipelines
  // support text prompts. If we introduce pipelines that truly do not support
  // text in the future we can add explicit metadata for that.
  const supportsVideo = category === "video-input";
  const supportsText = true;

  const requiresVideoInVideoMode = category === "video-input" && supportsVideo;

  const baseResolution = getDefaultResolution(id);

  let showNoiseControlsInText = false;
  let showNoiseControlsInVideo = false;

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

  return {
    id,
    category,
    nativeMode,
    supportsVideo,
    supportsText,
    requiresVideoInVideoMode,
    defaultResolutionByMode: {
      text: baseResolution,
      video: baseResolution,
    },
    showNoiseControlsInText,
    showNoiseControlsInVideo,
  };
}
