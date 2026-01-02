// Pipeline IDs are dynamic - any string returned from backend is valid
export type PipelineId = string;

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
  // Spout settings
  spoutReceiver?: {
    enabled: boolean;
    name: string;
  };
  spoutSender?: {
    enabled: boolean;
    name: string;
  };
  // VACE-specific settings
  vaceEnabled?: boolean;
  refImages?: string[];
  vaceContextScale?: number;
  // Depth preprocessor settings
  depthPreprocessor?: boolean;
  depthPreprocessorEncoder?: "vits" | "vitb" | "vitl";
  depthPreprocessorMode?: "depth_only";
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
  supportsPrompts?: boolean;
  defaultTemporalInterpolationMethod?: "linear" | "slerp";
  defaultTemporalInterpolationSteps?: number;
  supportsLoRA?: boolean;
  supportsVACE?: boolean;

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;

  // UI capabilities - tells frontend what controls to show
  supportsCacheManagement?: boolean;
  supportsKvCacheBias?: boolean;
  supportsQuantization?: boolean;
  minDimension?: number;
  recommendedQuantizationVramThreshold?: number | null;
}

export interface DownloadProgress {
  is_downloading: boolean;
  percentage: number;
  current_artifact: string | null;
}

export interface ModelStatusResponse {
  downloaded: boolean;
  progress: DownloadProgress | null;
}
