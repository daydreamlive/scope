import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getDefaultDenoisingSteps(pipelineId: PipelineId): number[] {
  if (pipelineId === "longlive") {
    return [1000, 750, 500, 250];
  } else if (pipelineId === "krea-realtime-video") {
    return [1000, 750, 500, 250];
  } else if (pipelineId === "streamdiffusionv2") {
    return [700, 500];
  }
  return [700, 500]; // Default fallback
}

export function getDefaultResolution(pipelineId: PipelineId): {
  height: number;
  width: number;
} {
  if (pipelineId === "longlive") {
    return { height: 320, width: 576 };
  } else if (pipelineId === "krea-realtime-video") {
    return { height: 320, width: 576 };
  } else if (pipelineId === "streamdiffusionv2") {
    return { height: 512, width: 512 };
  }
  return { height: 320, width: 576 }; // Default fallback
}

/**
 * Gets the scale factor that resolution must be divisible by for a given pipeline.
 * Returns null if the pipeline doesn't require resolution adjustment.
 */
export function getResolutionScaleFactor(
  pipelineId: PipelineId
): number | null {
  if (
    pipelineId === "longlive" ||
    pipelineId === "streamdiffusionv2" ||
    pipelineId === "krea-realtime-video"
  ) {
    // VAE downsample (8) * patch embedding downsample (2) = 16
    return 16;
  }
  return null;
}

/**
 * Adjusts resolution to be divisible by the required scale factor for the pipeline.
 * Returns the adjusted resolution and whether it was changed.
 */
export function adjustResolutionForPipeline(
  pipelineId: PipelineId,
  resolution: { height: number; width: number }
): {
  resolution: { height: number; width: number };
  wasAdjusted: boolean;
} {
  const scaleFactor = getResolutionScaleFactor(pipelineId);
  if (!scaleFactor) {
    return { resolution, wasAdjusted: false };
  }

  const adjustedHeight =
    Math.round(resolution.height / scaleFactor) * scaleFactor;
  const adjustedWidth =
    Math.round(resolution.width / scaleFactor) * scaleFactor;

  const wasAdjusted =
    adjustedHeight !== resolution.height || adjustedWidth !== resolution.width;

  return {
    resolution: { height: adjustedHeight, width: adjustedWidth },
    wasAdjusted,
  };
}
