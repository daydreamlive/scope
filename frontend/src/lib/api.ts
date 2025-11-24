import type { LoRAConfig } from "../types";

// Base API configuration
// In development: always use empty string to route through Vite proxy (avoids CORS)
// In production: use VITE_SCOPE_API_URL if set, otherwise empty (for same-origin)
// The Vite proxy (configured in vite.config.ts) handles forwarding to the correct backend
const API_BASE_URL = import.meta.env.DEV
  ? "" // Always use proxy in dev to avoid CORS issues
  : import.meta.env.VITE_SCOPE_API_URL || "";

// Auth token (optional)
// Set VITE_SCOPE_API_AUTH_TOKEN environment variable to add Authorization header
// Note: Vite requires the VITE_ prefix for environment variables exposed to client code
const API_AUTH_TOKEN = import.meta.env.VITE_SCOPE_API_AUTH_TOKEN;

// Debug: Log configuration (only in dev mode)
if (import.meta.env.DEV) {
  console.log("[API] Using relative URLs (routing through Vite proxy)");
  console.log(
    "[API] Proxy target:",
    import.meta.env.VITE_SCOPE_API_URL || "http://localhost:8000"
  );
  if (API_AUTH_TOKEN) {
    console.log(
      "[API] Authorization token loaded (length:",
      API_AUTH_TOKEN.length,
      ")"
    );
  } else {
    console.warn(
      "[API] No authorization token found. Set VITE_SCOPE_API_AUTH_TOKEN environment variable."
    );
  }
}

// Helper function to get headers with authorization
const getHeaders = (): HeadersInit => {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  // Only add Authorization header if token is provided
  if (API_AUTH_TOKEN) {
    headers.Authorization = `Bearer ${API_AUTH_TOKEN}`;
  }

  return headers;
};

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
}

export interface LongLiveLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
  quantization?: "fp8_e4m3fn" | null;
  loras?: LoRAConfig[];
  lora_merge_mode?: "permanent_merge" | "runtime_peft";
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

export const sendWebRTCOffer = async (
  data: WebRTCOfferRequest
): Promise<RTCSessionDescriptionInit> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/webrtc/offer`, {
    method: "POST",
    headers: getHeaders(),
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

export const loadPipeline = async (
  data: PipelineLoadRequest = {}
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/pipeline/load`, {
    method: "POST",
    headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/api/v1/pipeline/status`, {
    method: "GET",
    headers: getHeaders(),
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
    `${API_BASE_URL}/api/v1/models/status?pipeline_id=${pipelineId}`,
    {
      method: "GET",
      headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/api/v1/models/download`, {
    method: "POST",
    headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/api/v1/hardware/info`, {
    method: "GET",
    headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/api/v1/logs/current`, {
    method: "GET",
    headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/api/v1/lora/list`, {
    method: "GET",
    headers: getHeaders(),
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
