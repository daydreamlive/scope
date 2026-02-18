import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
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
    pipelineId === "krea-realtime-video" ||
    pipelineId === "reward-forcing"
  ) {
    // VAE downsample (8) * patch embedding downsample (2) = 16
    return 16;
  }
  return null;
}

/**
 * Scales a source resolution down to fit within a pixel budget while preserving
 * aspect ratio. Dimensions are rounded to the nearest multiple of scaleFactor.
 * If the source already fits, it is returned as-is (rounded to scaleFactor).
 */
export function fitResolutionToPixelBudget(
  sourceWidth: number,
  sourceHeight: number,
  maxPixels: number,
  scaleFactor: number
): { width: number; height: number } {
  const sourcePixels = sourceWidth * sourceHeight;
  if (sourcePixels <= maxPixels) {
    return {
      width: Math.max(
        scaleFactor,
        Math.round(sourceWidth / scaleFactor) * scaleFactor
      ),
      height: Math.max(
        scaleFactor,
        Math.round(sourceHeight / scaleFactor) * scaleFactor
      ),
    };
  }
  const scale = Math.sqrt(maxPixels / sourcePixels);
  return {
    width: Math.max(
      scaleFactor,
      Math.round((sourceWidth * scale) / scaleFactor) * scaleFactor
    ),
    height: Math.max(
      scaleFactor,
      Math.round((sourceHeight * scale) / scaleFactor) * scaleFactor
    ),
  };
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
