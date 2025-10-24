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
