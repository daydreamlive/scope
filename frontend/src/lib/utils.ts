import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineId } from "../types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getDefaultDenoisingSteps(pipelineId: PipelineId): number[] {
  if (pipelineId === "longlive") {
    return [1000, 750, 500, 250];
  } else if (pipelineId === "streamdiffusionv2") {
    return [700, 500];
  }
  return [700, 500]; // Default fallback
}
