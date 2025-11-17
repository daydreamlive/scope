export type PipelineId =
  | "streamdiffusionv2-1.3b"
  | "streamdiffusionv2-14b"
  | "passthrough"
  | "vod"
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

export interface SettingsState {
  pipelineId: PipelineId;
  resolution?: {
    height: number;
    width: number;
  };
  seed?: number;
  denoisingSteps?: number[];
  noiseScale?: number;
  noiseController?: boolean;
  manageCache?: boolean;
  quantization?: "fp8_e4m3fn" | null;
  kvCacheAttentionBias?: number;
  paused?: boolean;
  loras?: LoRAConfig[];
  loraMergeStrategy?: LoraMergeStrategy;
}

export type PipelineCategory = "video-input" | "no-video-input";

export interface PipelineInfo {
  name: string;
  about: string;
  projectUrl?: string;
  modified?: boolean;
  category: PipelineCategory;
}
