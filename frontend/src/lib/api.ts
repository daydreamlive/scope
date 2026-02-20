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
  data: PipelineLoadRequest
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
): Promise<ModelStatusResponse> => {
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
  spout_available: boolean;
  ndi_available: boolean;
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

// Input sources API

export interface InputSourceType {
  source_id: string;
  source_name: string;
  source_description: string;
  available: boolean;
}

export interface InputSourceTypesResponse {
  input_sources: InputSourceType[];
}

export interface DiscoveredSource {
  name: string;
  identifier: string;
  metadata: Record<string, unknown> | null;
}

export interface DiscoveredSourcesResponse {
  source_type: string;
  sources: DiscoveredSource[];
}

export const getInputSources = async (): Promise<InputSourceTypesResponse> => {
  const response = await fetch("/api/v1/input-sources", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Input sources failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  return response.json();
};

export const getInputSourceSources = async (
  sourceType: string,
  timeoutMs = 5000
): Promise<DiscoveredSourcesResponse> => {
  const response = await fetch(
    `/api/v1/input-sources/${sourceType}/sources?timeout_ms=${timeoutMs}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Discover sources failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  return response.json();
};

export interface InputSourceResolution {
  width: number;
  height: number;
}

export const getInputSourceResolution = async (
  sourceType: string,
  identifier: string,
  timeoutMs = 5000
): Promise<InputSourceResolution> => {
  const response = await fetch(
    `/api/v1/input-sources/${sourceType}/sources/${encodeURIComponent(identifier)}/resolution?timeout_ms=${timeoutMs}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Probe resolution failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  return response.json();
};

export const getInputSourceStreamUrl = (
  sourceType: string,
  identifier: string,
  fps = 2
): string =>
  `/api/v1/input-sources/${sourceType}/sources/${encodeURIComponent(identifier)}/stream?fps=${fps}`;

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
  const response = await fetch("/api/v1/loras", {
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

export interface LoRAInstallRequest {
  url: string;
  filename?: string;
}

export interface LoRAInstallResponse {
  message: string;
  file: LoRAFileInfo;
}

export const installLoRAFile = async (
  data: LoRAInstallRequest
): Promise<LoRAInstallResponse> => {
  const response = await fetch("/api/v1/loras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Install LoRA failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface LoRADeleteResponse {
  success: boolean;
  message: string;
}

export const deleteLoRAFile = async (
  name: string
): Promise<LoRADeleteResponse> => {
  const response = await fetch(`/api/v1/loras/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
};

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

export const listAssets = async (
  type?: "image" | "video"
): Promise<AssetsResponse> => {
  const url = type ? `/api/v1/assets?type=${type}` : "/api/v1/assets";
  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `List assets failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const uploadAsset = async (file: File): Promise<AssetFileInfo> => {
  const fileContent = await file.arrayBuffer();
  const filename = encodeURIComponent(file.name);

  const response = await fetch(`/api/v1/assets?filename=${filename}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
    },
    body: fileContent,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Upload asset failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const getAssetUrl = (assetPath: string): string => {
  // The backend returns full absolute paths, but we need to extract the relative path
  // from the assets directory for the serving endpoint
  // Example: C:\Users\...\assets\myimage.png -> myimage.png
  // or: C:\Users\...\assets\subfolder\myimage.png -> subfolder/myimage.png

  const pathParts = assetPath.split(/[/\\]/);
  const assetsIndex = pathParts.findIndex(
    part => part === "assets" || part === ".daydream-scope"
  );

  if (assetsIndex >= 0 && assetsIndex < pathParts.length - 1) {
    // Find the assets directory and take everything after it
    const assetsPos = pathParts.findIndex(part => part === "assets");
    if (assetsPos >= 0) {
      const relativePath = pathParts.slice(assetsPos + 1).join("/");
      return `/api/v1/assets/${relativePath}`;
    }
  }

  // Fallback: just use the filename
  const filename = pathParts[pathParts.length - 1];
  return `/api/v1/assets/${encodeURIComponent(filename)}`;
};

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

// Plugin types
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

export interface FailedPluginInfo {
  package_name: string;
  entry_point_name: string;
  error_type: string;
  error_message: string;
}

export interface PluginListResponse {
  plugins: PluginInfo[];
  total: number;
  failed_plugins: FailedPluginInfo[];
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

// Helper to extract user-friendly error message from API response
const extractErrorDetail = async (response: Response): Promise<string> => {
  try {
    const errorJson = await response.json();
    // FastAPI HTTPException returns { detail: "message" }
    if (errorJson.detail) {
      return typeof errorJson.detail === "string"
        ? errorJson.detail
        : JSON.stringify(errorJson.detail);
    }
    return JSON.stringify(errorJson);
  } catch {
    // If JSON parsing fails, fall back to text
    return response.statusText || "Unknown error";
  }
};

export const listPlugins = async (): Promise<PluginListResponse> => {
  const response = await fetch("/api/v1/plugins");
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }
  return response.json();
};

export const installPlugin = async (
  request: PluginInstallRequest
): Promise<PluginInstallResponse> => {
  const response = await fetch("/api/v1/plugins", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }
  return response.json();
};

export const uninstallPlugin = async (
  name: string
): Promise<PluginUninstallResponse> => {
  const response = await fetch(`/api/v1/plugins/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }
  return response.json();
};

export const restartServer = async (): Promise<number | null> => {
  // Get current server start time before triggering restart
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
        // If we have an old start time, wait for it to change
        if (oldStartTime === null || data.server_start_time !== oldStartTime) {
          // New server is up
          return;
        }
        // Same server still running, keep waiting
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

export async function getServerInfo(): Promise<ServerInfo> {
  const response = await fetch("/health");
  if (!response.ok) {
    throw new Error("Failed to fetch server info");
  }
  const data: HealthResponse = await response.json();
  return { version: data.version, gitCommit: data.git_commit };
}

// API Key management types and functions

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

export const getApiKeys = async (): Promise<{ keys: ApiKeyInfo[] }> => {
  const response = await fetch("/api/v1/keys", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Get API keys failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  return response.json();
};

export const setApiKey = async (
  serviceId: string,
  value: string
): Promise<ApiKeySetResponse> => {
  const response = await fetch(
    `/api/v1/keys/${encodeURIComponent(serviceId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    }
  );

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
};

export const deleteApiKey = async (
  serviceId: string
): Promise<ApiKeyDeleteResponse> => {
  const response = await fetch(
    `/api/v1/keys/${encodeURIComponent(serviceId)}`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
};

export const downloadRecording = async (sessionId: string): Promise<void> => {
  if (!sessionId) {
    throw new Error("Session ID is required to download recording");
  }

  const response = await fetch(`/api/v1/recordings/${sessionId}`, {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Download recording failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  // Get the blob and trigger download
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
