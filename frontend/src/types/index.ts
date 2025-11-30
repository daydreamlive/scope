import type { InputMode } from "../constants/modes";

/**
 * Null/Undefined Convention:
 * - undefined: Not loaded yet, not set, or using default
 * - null: Explicitly disabled or not applicable
 */

export type PipelineId =
  | "streamdiffusionv2"
  | "passthrough"
  | "longlive"
  | "krea-realtime-video";

export interface SystemMetrics {
  cpu: number;
  gpu: number;
  systemRAM: number;
  vram: number;
  fps: number;
  latency: number;
}

export interface StreamStatus {
  status: string;
}

export interface PromptData {
  prompt: string;
  isProcessing: boolean;
}

export type LoraMergeStrategy = "permanent_merge" | "runtime_peft";

export interface LoRAConfig {
  id: string;
  path: string;
  scale: number;
}

/**
 * Settings state for stream configuration.
 *
 * Convention for optional fields:
 * - undefined = not loaded yet / not set by user (waiting for schema)
 * - null = explicitly not applicable / disabled for current mode
 *
 * Fields with | null union: noiseScale, noiseController, quantization
 * (These can be disabled/not applicable depending on mode)
 *
 * Fields without | null: inputMode, resolution, seed, denoisingSteps, etc.
 * (These are always applicable, just may not be loaded yet)
 */
export interface SettingsState {
  pipelineId: PipelineId;
  inputMode?: InputMode;
  resolution?: {
    height: number;
    width: number;
  };
  seed?: number;
  denoisingSteps?: number[];
  noiseScale?: number | null;
  noiseController?: boolean | null;
  manageCache?: boolean;
  quantization?: "fp8_e4m3fn" | null;
  kvCacheAttentionBias?: number;
  paused?: boolean;
  loras?: LoRAConfig[];
  loraMergeStrategy?: LoraMergeStrategy;
  samplerType?: string;
}
