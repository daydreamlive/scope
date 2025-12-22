export type PipelineId =
  | "streamdiffusionv2"
  | "passthrough"
  | "longlive"
  | "krea-realtime-video"
  | "reward-forcing"
  | "personalive";

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
}

// UI capability flags - controls which UI elements are shown for each pipeline
export interface PipelineUICapabilities {
  showTimeline?: boolean; // Show prompt timeline (default: true)
  showPromptInput?: boolean; // Show prompt input controls (default: true)
  showResolutionControl?: boolean; // Show resolution width/height controls
  showSeedControl?: boolean; // Show seed control
  showDenoisingSteps?: boolean; // Show denoising steps slider
  showCacheManagement?: boolean; // Show cache management toggle and reset
  showQuantization?: boolean; // Show quantization selector
  showKvCacheAttentionBias?: boolean; // Show KV cache attention bias slider (krea-realtime-video)
  showNoiseControls?: boolean; // Show noise scale and controller (video mode)
  canChangeReferenceWhileStreaming?: boolean; // Allow changing reference image during stream
}

export interface PipelineInfo {
  name: string;
  about: string;
  projectUrl?: string;
  docsUrl?: string;
  defaultPrompt?: string;
  estimatedVram?: number;
  requiresModels?: boolean;
  defaultTemporalInterpolationMethod?: "linear" | "slerp";
  defaultTemporalInterpolationSteps?: number;
  supportsLoRA?: boolean;
  supportsVACE?: boolean;
  supportsPrompts?: boolean;
  requiresReferenceImage?: boolean; // Whether this pipeline requires a reference image (e.g. PersonaLive)
  referenceImageDescription?: string; // Description of what the reference image is for

  // Multi-mode support
  supportedModes: InputMode[];
  defaultMode: InputMode;

  // UI capabilities - controls which settings/controls are shown
  ui?: PipelineUICapabilities;
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
