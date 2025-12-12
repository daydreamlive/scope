export type PipelineId =
  | "streamdiffusionv2"
  | "passthrough"
  | "longlive"
  | "longlive_vace"
  | "krea-realtime-video"
  | "reward-forcing";

// Input mode for pipeline operation
export type InputMode = "text" | "video";

// WebRTC ICE server configuration
export interface IceServerConfig {
  urls: string | string[];
  username?: string;
  credential?: string;
}

export interface IceServersResponse {
  iceServers: IceServerConfig[];
}

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
  mergeMode?: LoraMergeStrategy;
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
  // Track current input mode (text vs video)
  inputMode?: InputMode;
  // VACE-specific settings
  refImages?: string[];
  vaceContextScale?: number;
}

export interface PipelineInfo {
  name: string;
  about: string;
  projectUrl?: string;
  docsUrl?: string;
  modified?: boolean;
  defaultPrompt?: string;
  estimatedVram?: number;
  requiresModels?: boolean;
  defaultTemporalInterpolationMethod?: "linear" | "slerp";
  defaultTemporalInterpolationSteps?: number;
  supportsLoRA?: boolean;

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;
}
