import type { LoRAConfig, IceServersResponse } from "../types";

export interface PromptItem {
  text: string;
  weight: number;
}

export interface PromptTransition {
  target_prompts: PromptItem[];
  num_steps?: number; // Default: 4
  temporal_interpolation_method?: "linear" | "slerp"; // Default: linear
}

export interface WebRTCOfferRequest {
  sdp?: string;
  type?: string;
  initialParameters?: {
    prompts?: string[] | PromptItem[];
    prompt_interpolation_method?: "linear" | "slerp";
    transition?: PromptTransition;
    denoising_step_list?: number[];
    noise_scale?: number;
    noise_controller?: boolean;
    manage_cache?: boolean;
    kv_cache_attention_bias?: number;
    ref_images?: string[];
    vace_context_scale?: number;
  };
}

export interface PipelineLoadParams {
  // Base interface for pipeline load parameters
  [key: string]: unknown;
}

export interface PassthroughLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
}

export interface StreamDiffusionV2LoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
  quantization?: "fp8_e4m3fn" | null;
  loras?: LoRAConfig[];
  lora_merge_mode?: "permanent_merge" | "runtime_peft";
  // VACE (optional reference image conditioning for text mode)
  ref_images?: string[];
  vace_context_scale?: number;
}

export interface LongLiveLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
  quantization?: "fp8_e4m3fn" | null;
  loras?: LoRAConfig[];
  lora_merge_mode?: "permanent_merge" | "runtime_peft";
  // VACE (optional reference image conditioning)
  ref_images?: string[];
  vace_context_scale?: number;
}

export interface KreaRealtimeVideoLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
  quantization?: "fp8_e4m3fn" | null;
  loras?: LoRAConfig[];
  lora_merge_mode?: "permanent_merge" | "runtime_peft";
}

export interface PipelineLoadRequest {
  pipeline_id?: string;
  load_params?:
    | PassthroughLoadParams
    | StreamDiffusionV2LoadParams
    | LongLiveLoadParams
    | KreaRealtimeVideoLoadParams
    | null;
}

export interface PipelineStatusResponse {
  status: "not_loaded" | "loading" | "loaded" | "error";
  pipeline_id?: string;
  load_params?: Record<string, unknown>;
  // Optional list of loaded LoRA adapters, provided by backend when available.
  loaded_lora_adapters?: { path: string; scale: number }[];
  error?: string;
}

export const getIceServers = async (): Promise<IceServersResponse> => {
  const response = await fetch("/api/v1/webrtc/ice-servers", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Get ICE servers failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface WebRTCOfferResponse {
  sdp: string;
  type: string;
  sessionId: string;
}

export const sendWebRTCOffer = async (
  data: WebRTCOfferRequest
): Promise<WebRTCOfferResponse> => {
  const response = await fetch("/api/v1/webrtc/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `WebRTC offer failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const sendIceCandidates = async (
  sessionId: string,
  candidates: RTCIceCandidate | RTCIceCandidate[]
): Promise<void> => {
  const candidateArray = Array.isArray(candidates) ? candidates : [candidates];

  const response = await fetch(`/api/v1/webrtc/offer/${sessionId}`, {
    method: "PATCH",
    // TODO: Use Content-Type 'application/trickle-ice-sdpfrag'
    // once backend supports it
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidates: candidateArray.map(c => ({
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      })),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Send ICE candidate failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }
};

export const loadPipeline = async (
  data: PipelineLoadRequest = {}
): Promise<{ message: string }> => {
  const response = await fetch("/api/v1/pipeline/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Pipeline load failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const getPipelineStatus = async (): Promise<PipelineStatusResponse> => {
  const response = await fetch("/api/v1/pipeline/status", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(30000), // 30 second timeout per request
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Pipeline status failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const checkModelStatus = async (
  pipelineId: string
): Promise<{ downloaded: boolean }> => {
  const response = await fetch(
    `/api/v1/models/status?pipeline_id=${pipelineId}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Model status check failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const downloadPipelineModels = async (
  pipelineId: string
): Promise<{ message: string }> => {
  const response = await fetch("/api/v1/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pipeline_id: pipelineId }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Model download failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface HardwareInfoResponse {
  vram_gb: number | null;
}

export const getHardwareInfo = async (): Promise<HardwareInfoResponse> => {
  const response = await fetch("/api/v1/hardware/info", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Hardware info failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const fetchCurrentLogs = async (): Promise<string> => {
  const response = await fetch("/api/v1/logs/current", {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Fetch logs failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const logsText = await response.text();
  return logsText;
};

export interface LoRAFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
}

export interface LoRAFilesResponse {
  lora_files: LoRAFileInfo[];
}

export const listLoRAFiles = async (): Promise<LoRAFilesResponse> => {
  const response = await fetch("/api/v1/lora/list", {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `List LoRA files failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface ImageFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
}

export interface ImageFilesResponse {
  image_files: ImageFileInfo[];
}

export const listImageFiles = async (): Promise<ImageFilesResponse> => {
  const response = await fetch("/api/v1/images/list", {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `List image files failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

// Pipeline schema types - matches output of get_schema_with_metadata()
export interface PipelineSchemaProperty {
  type?: string;
  default?: unknown;
  description?: string;
  // JSON Schema fields
  minimum?: number;
  maximum?: number;
  items?: unknown;
  anyOf?: unknown[];
}

export interface PipelineConfigSchema {
  type: string;
  properties: Record<string, PipelineSchemaProperty>;
  required?: string[];
  title?: string;
}

// Mode-specific default overrides
export interface ModeDefaults {
  height?: number;
  width?: number;
  denoising_steps?: number[];
  noise_scale?: number | null;
  noise_controller?: boolean | null;
}

export interface PipelineSchemaInfo {
  id: string;
  name: string;
  description: string;
  version: string;
  config_schema: PipelineConfigSchema;
  // Mode support - comes from config class
  supported_modes: ("text" | "video")[];
  default_mode: "text" | "video";
  // Mode-specific default overrides (optional)
  mode_defaults?: Record<"text" | "video", ModeDefaults>;
}

export interface PipelineSchemasResponse {
  pipelines: Record<string, PipelineSchemaInfo>;
}

export const getPipelineSchemas =
  async (): Promise<PipelineSchemasResponse> => {
    const response = await fetch("/api/v1/pipelines/schemas", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Get pipeline schemas failed: ${response.status} ${response.statusText}: ${errorText}`
      );
    }

    const result = await response.json();
    return result;
  };
