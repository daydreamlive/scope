import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PipelineDefaultsResponse, PipelineModeConfig } from "./api";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getCurrentModeConfig(
  defaults: PipelineDefaultsResponse | null
): PipelineModeConfig | null {
  if (!defaults) return null;

  const mode = defaults.native_generation_mode;
  const modeConfig = defaults.modes[mode];

  if (!modeConfig) {
    console.error(
      `getCurrentModeConfig: Native mode '${mode}' not found in pipeline defaults. Available modes:`,
      Object.keys(defaults.modes)
    );
    return null;
  }

  return modeConfig;
}
