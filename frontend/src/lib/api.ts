import type { IceServersResponse, ModelStatusResponse } from "../types";

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
    vace_ref_images?: string[];
    vace_context_scale?: number;
    pipeline_ids?: string[];
    images?: string[];
  };
}

export interface PipelineLoadParams {
  // Base interface for pipeline load parameters
  [key: string]: unknown;
}

// Generic load params - accepts any key-value pairs based on pipeline config
export type PipelineLoadParamsGeneric = Record<string, unknown>;

export interface PipelineLoadRequest {
  pipeline_ids: string[];
  load_params?: PipelineLoadParamsGeneric | null;
}

export interface PipelineStatusResponse {
  status: "not_loaded" | "loading" | "loaded" | "error";
  pipeline_id?: string;
  load_params?: Record<string, unknown>;
  // Optional list of loaded LoRA adapters, provided by backend when available.
  loaded_lora_adapters?: { path: string; scale: number }[];
  error?: string;
}

// ---------------------------------------------------------------------------
// Shared fetch helpers
// ---------------------------------------------------------------------------

const extractErrorDetail = async (response: Response): Promise<string> => {
  try {
    const text = await response.text();
    try {
      const errorJson = JSON.parse(text);
      if (errorJson.detail) {
        return typeof errorJson.detail === "string"
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
      return text;
    } catch {
      return text || `${response.status} ${response.statusText}`;
    }
  } catch {
    return `${response.status} ${response.statusText}`;
  }
};

async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }
  return response.json();
}

async function apiFetchRaw(
  url: string,
  options?: RequestInit
): Promise<Response> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }
  return response;
}

const jsonHeaders = { "Content-Type": "application/json" } as const;

// ---------------------------------------------------------------------------
// WebRTC
// ---------------------------------------------------------------------------

export interface WebRTCOfferResponse {
  sdp: string;
  type: string;
  sessionId: string;
}

export const getIceServers = (): Promise<IceServersResponse> =>
  apiFetch("/api/v1/webrtc/ice-servers", { headers: jsonHeaders });

export const sendWebRTCOffer = (
  data: WebRTCOfferRequest
): Promise<WebRTCOfferResponse> =>
  apiFetch("/api/v1/webrtc/offer", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(data),
  });

export const sendIceCandidates = async (
  sessionId: string,
  candidates: RTCIceCandidate | RTCIceCandidate[]
): Promise<void> => {
  const candidateArray = Array.isArray(candidates) ? candidates : [candidates];
  await apiFetchRaw(`/api/v1/webrtc/offer/${sessionId}`, {
    method: "PATCH",
    headers: jsonHeaders,
    body: JSON.stringify({
      candidates: candidateArray.map(c => ({
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      })),
    }),
  });
};

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

export const loadPipeline = (
  data: PipelineLoadRequest
): Promise<{ message: string }> =>
  apiFetch("/api/v1/pipeline/load", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(data),
  });

export const getPipelineStatus = (): Promise<PipelineStatusResponse> =>
  apiFetch("/api/v1/pipeline/status", {
    headers: jsonHeaders,
    signal: AbortSignal.timeout(30000),
  });

// UI metadata from pipeline schema (json_schema_extra on fields)
export interface SchemaFieldUI {
  category?: string;
  order?: number;
  component?: string;
  modes?: ("text" | "video")[];
  /** If true, field is a load param (disabled when streaming); if false, runtime param (editable when streaming). Omit = treated as load param. */
  is_load_param?: boolean;
}

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
  enum?: unknown[];
  $ref?: string;
  /** UI hints from backend (Field json_schema_extra) */
  ui?: SchemaFieldUI;
}

export interface PipelineConfigSchema {
  type: string;
  properties: Record<string, PipelineSchemaProperty>;
  required?: string[];
  title?: string;
  $defs?: Record<string, { enum?: unknown[] }>;
}

// Mode-specific default overrides
export interface ModeDefaults {
  height?: number;
  width?: number;
  denoising_steps?: number[];
  noise_scale?: number | null;
  noise_controller?: boolean | null;
  default_temporal_interpolation_steps?: number;
}

export interface PipelineSchemaInfo {
  id: string;
  name: string;
  description: string;
  version: string;
  docs_url: string | null;
  estimated_vram_gb: number | null;
  requires_models: boolean;
  supports_lora: boolean;
  supports_vace: boolean;
  usage: string[];
  // Pipeline config schema
  config_schema: PipelineConfigSchema;
  // Mode support - comes from config class
  supported_modes: ("text" | "video")[];
  default_mode: "text" | "video";
  // Prompt and temporal interpolation support
  supports_prompts: boolean;
  default_temporal_interpolation_method: "linear" | "slerp" | null;
  default_temporal_interpolation_steps: number | null;
  default_spatial_interpolation_method: "linear" | "slerp" | null;
  // Mode-specific default overrides (optional)
  mode_defaults?: Record<"text" | "video", ModeDefaults>;
  // UI capabilities
  supports_cache_management: boolean;
  supports_kv_cache_bias: boolean;
  supports_quantization: boolean;
  min_dimension: number;
  recommended_quantization_vram_threshold: number | null;
  modified: boolean;
  plugin_name: string | null;
}

export interface PipelineSchemasResponse {
  pipelines: Record<string, PipelineSchemaInfo>;
}

export const getPipelineSchemas = (): Promise<PipelineSchemasResponse> =>
  apiFetch("/api/v1/pipelines/schemas", { headers: jsonHeaders });

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export const checkModelStatus = (
  pipelineId: string
): Promise<ModelStatusResponse> =>
  apiFetch(`/api/v1/models/status?pipeline_id=${pipelineId}`, {
    headers: jsonHeaders,
  });

export const downloadPipelineModels = (
  pipelineId: string
): Promise<{ message: string }> =>
  apiFetch("/api/v1/models/download", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ pipeline_id: pipelineId }),
  });

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

export interface HardwareInfoResponse {
  vram_gb: number | null;
  spout_available: boolean;
}

export const getHardwareInfo = (): Promise<HardwareInfoResponse> =>
  apiFetch("/api/v1/hardware/info", { headers: jsonHeaders });

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------

export const fetchCurrentLogs = async (): Promise<string> => {
  const response = await apiFetchRaw("/api/v1/logs/current");
  return response.text();
};

// ---------------------------------------------------------------------------
// LoRA
// ---------------------------------------------------------------------------

export interface LoRAFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
}

export interface LoRAFilesResponse {
  lora_files: LoRAFileInfo[];
}

export const listLoRAFiles = (): Promise<LoRAFilesResponse> =>
  apiFetch("/api/v1/lora/list");

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------

export interface AssetFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
  type: string; // "image" or "video"
  created_at: number; // Unix timestamp
}

export interface AssetsResponse {
  assets: AssetFileInfo[];
}

export const listAssets = (type?: "image" | "video"): Promise<AssetsResponse> =>
  apiFetch(type ? `/api/v1/assets?type=${type}` : "/api/v1/assets");

export const uploadAsset = async (file: File): Promise<AssetFileInfo> => {
  const fileContent = await file.arrayBuffer();
  const filename = encodeURIComponent(file.name);
  return apiFetch(`/api/v1/assets?filename=${filename}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: fileContent,
  });
};

export const getAssetUrl = (assetPath: string): string => {
  // The backend returns full absolute paths, but we need to extract the relative path
  // from the assets directory for the serving endpoint
  const pathParts = assetPath.split(/[/\\]/);
  const assetsIndex = pathParts.findIndex(
    part => part === "assets" || part === ".daydream-scope"
  );

  if (assetsIndex >= 0 && assetsIndex < pathParts.length - 1) {
    const assetsPos = pathParts.findIndex(part => part === "assets");
    if (assetsPos >= 0) {
      const relativePath = pathParts.slice(assetsPos + 1).join("/");
      return `/api/v1/assets/${relativePath}`;
    }
  }

  const filename = pathParts[pathParts.length - 1];
  return `/api/v1/assets/${encodeURIComponent(filename)}`;
};

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

export interface PluginPipelineInfo {
  pipeline_id: string;
  pipeline_name: string;
}

export interface PluginInfo {
  name: string;
  version: string | null;
  author: string | null;
  description: string | null;
  source: "pypi" | "git" | "local";
  editable: boolean;
  editable_path: string | null;
  pipelines: PluginPipelineInfo[];
  latest_version: string | null;
  update_available: boolean | null;
  package_spec: string | null;
}

export interface PluginListResponse {
  plugins: PluginInfo[];
  total: number;
}

export interface PluginInstallRequest {
  package: string;
  editable?: boolean;
  upgrade?: boolean;
  force?: boolean;
  pre?: boolean;
}

export interface PluginInstallResponse {
  success: boolean;
  message: string;
  plugin: PluginInfo | null;
}

export interface PluginUninstallResponse {
  success: boolean;
  message: string;
  unloaded_pipelines: string[];
}

export const listPlugins = (): Promise<PluginListResponse> =>
  apiFetch("/api/v1/plugins");

export const installPlugin = (
  request: PluginInstallRequest
): Promise<PluginInstallResponse> =>
  apiFetch("/api/v1/plugins", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(request),
  });

export const uninstallPlugin = (
  name: string
): Promise<PluginUninstallResponse> =>
  apiFetch(`/api/v1/plugins/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

export const restartServer = async (): Promise<number | null> => {
  let oldStartTime: number | null = null;
  try {
    const response = await fetch(`/health?_t=${Date.now()}`);
    if (response.ok) {
      const data = await response.json();
      oldStartTime = data.server_start_time;
    }
  } catch {
    // Ignore
  }

  try {
    await fetch("/api/v1/restart", { method: "POST" });
  } catch {
    // Expected - server shuts down and connection is lost
  }

  return oldStartTime;
};

export const waitForServer = async (
  oldStartTime: number | null,
  maxAttempts = 30,
  delayMs = 1000
): Promise<void> => {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch(`/health?_t=${Date.now()}`);
      if (response.ok) {
        const data = await response.json();
        if (oldStartTime === null || data.server_start_time !== oldStartTime) {
          return;
        }
      }
    } catch {
      // Server not ready yet
    }
    await new Promise(r => setTimeout(r, delayMs));
  }
  throw new Error("Server did not restart in time");
};

export interface HealthResponse {
  status: string;
  timestamp: string;
  server_start_time: number;
  version: string;
  git_commit: string;
}

export interface ServerInfo {
  version: string;
  gitCommit: string;
}

export const getServerInfo = async (): Promise<ServerInfo> => {
  const data = await apiFetch<HealthResponse>("/health");
  return { version: data.version, gitCommit: data.git_commit };
};

// ---------------------------------------------------------------------------
// API Keys
// ---------------------------------------------------------------------------

export interface ApiKeyInfo {
  id: string;
  name: string;
  description: string;
  is_set: boolean;
  source: string | null;
  env_var: string | null;
  key_url: string | null;
}

export interface ApiKeySetResponse {
  success: boolean;
  message: string;
}

export interface ApiKeyDeleteResponse {
  success: boolean;
  message: string;
}

export const getApiKeys = (): Promise<{ keys: ApiKeyInfo[] }> =>
  apiFetch("/api/v1/keys", { headers: jsonHeaders });

export const setApiKey = (
  serviceId: string,
  value: string
): Promise<ApiKeySetResponse> =>
  apiFetch(`/api/v1/keys/${encodeURIComponent(serviceId)}`, {
    method: "PUT",
    headers: jsonHeaders,
    body: JSON.stringify({ value }),
  });

export const deleteApiKey = (
  serviceId: string
): Promise<ApiKeyDeleteResponse> =>
  apiFetch(`/api/v1/keys/${encodeURIComponent(serviceId)}`, {
    method: "DELETE",
  });

// ---------------------------------------------------------------------------
// Recordings
// ---------------------------------------------------------------------------

export const downloadRecording = async (sessionId: string): Promise<void> => {
  if (!sessionId) {
    throw new Error("Session ID is required to download recording");
  }

  const response = await apiFetchRaw(`/api/v1/recordings/${sessionId}`);

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `recording-${new Date().toISOString().split("T")[0]}.mp4`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
