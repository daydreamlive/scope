// Pipeline IDs are dynamic - any string returned from backend is valid
export type PipelineId = string;

// Input mode for pipeline operation
export type InputMode = "text" | "video";

// VAE type for model selection (dynamic from backend registry)
export type VaeType = string;

// Extension mode for FFLF (First-Frame-Last-Frame) feature
export type ExtensionMode = "firstframe" | "lastframe" | "firstlastframe";

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
  vaceUseInputVideo?: boolean;
  refImages?: string[];
  vaceContextScale?: number;
  // FFLF (First-Frame-Last-Frame) extension mode
  firstFrameImage?: string;
  lastFrameImage?: string;
  extensionMode?: ExtensionMode;
  // VAE type selection
  vaeType?: VaeType;
  // Preprocessors
  preprocessorIds?: string[];
  // Postprocessors
  postprocessorIds?: string[];
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
  defaultTemporalInterpolationMethod?: "linear" | "slerp" | null;
  defaultTemporalInterpolationSteps?: number | null;
  defaultSpatialInterpolationMethod?: "linear" | "slerp" | null;
  supportsLoRA?: boolean;
  supportsVACE?: boolean;
  usage?: string[];

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;

  // UI capabilities - tells frontend what controls to show
  supportsCacheManagement?: boolean;
  supportsKvCacheBias?: boolean;
  supportsQuantization?: boolean;
  minDimension?: number;
  recommendedQuantizationVramThreshold?: number | null;
  // Available VAE types from config schema enum (derived from vae_type field presence)
  vaeTypes?: string[];
  // Controller input support - presence of ctrl_input field in pipeline schema
  supportsControllerInput?: boolean;
  // Images input support - presence of images field in pipeline schema
  supportsImages?: boolean;
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
