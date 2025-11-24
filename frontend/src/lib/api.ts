import type { LoRAConfig } from "../types";

// Base API configuration
// In development: always use empty string to route through Vite proxy (avoids CORS)
// In production: use VITE_SCOPE_API_URL if set, otherwise empty (for same-origin)
// The Vite proxy (configured in vite.config.ts) handles forwarding to the correct backend
const API_BASE_URL = import.meta.env.DEV
  ? "" // Always use proxy in dev to avoid CORS issues
  : import.meta.env.VITE_SCOPE_API_URL || "";

// Reserve endpoint base URL (port 8080)
// In development: use empty string to route through Vite proxy
// In production: use VITE_SCOPE_RESERVE_URL if set, otherwise default to localhost:8080
const RESERVE_BASE_URL = import.meta.env.DEV
  ? "" // Use proxy in dev to avoid CORS issues
  : import.meta.env.VITE_SCOPE_RESERVE_URL || "http://localhost:8080";

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
  data: WebRTCOfferRequest,
  baseUrl?: string
): Promise<RTCSessionDescriptionInit> => {
  // Ensure proper URL construction
  let url: string;
  if (baseUrl) {
    // Remove trailing slashes from baseUrl
    const cleanBase = baseUrl.replace(/\/+$/, "");
    url = `${cleanBase}/api/v1/webrtc/offer`;
  } else {
    // Use default API_BASE_URL (empty string in dev, or configured URL)
    url = API_BASE_URL
      ? `${API_BASE_URL}/api/v1/webrtc/offer`
      : "/api/v1/webrtc/offer";
  }

  console.log("[sendWebRTCOffer] Using URL:", url, "baseUrl:", baseUrl);

  const response = await fetch(url, {
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

export interface ReserveResponse {
  host: string;
  port: string;
}

/**
 * Call the /reserve endpoint and keep the connection open using HTTP event streaming.
 * Returns the resolved host and port, an AbortController to close the connection,
 * and a promise that resolves when the reader finishes (for reference).
 */
export const callReserve = async (): Promise<{
  data: ReserveResponse;
  abortController: AbortController;
  readerPromise: Promise<void>;
}> => {
  const abortController = new AbortController();

  const response = await fetch(`${RESERVE_BASE_URL}/reserve`, {
    method: "GET",
    signal: abortController.signal,
    // Ensure the browser treats this as a streaming response
    cache: "no-cache",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Reserve endpoint failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  // Ensure we have a readable stream
  if (!response.body) {
    throw new Error("Response body is not readable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let reserveData: ReserveResponse | null = null;
  let jsonParsed = false;
  let resolveData: ((data: ReserveResponse) => void) | null = null;
  let rejectData: ((error: Error) => void) | null = null;

  // Promise that resolves when JSON is parsed
  const dataPromise = new Promise<ReserveResponse>((resolve, reject) => {
    resolveData = resolve;
    rejectData = reject;
  });

  // Function to continuously read from the stream to keep the connection alive
  // This MUST run continuously without gaps to prevent the browser from closing the connection
  const keepAliveReader = async (): Promise<void> => {
    try {
      // Read continuously without any delays
      while (!abortController.signal.aborted) {
        const { value, done } = await reader.read();

        if (done) {
          console.log("[callReserve] Stream ended by server");
          if (!jsonParsed && rejectData) {
            rejectData(new Error("Stream ended before JSON was received"));
          }
          break;
        }

        if (value) {
          // Decode the chunk
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // Parse JSON from the first line if not already parsed
          if (!jsonParsed) {
            const newlineIndex = buffer.indexOf("\n");
            if (newlineIndex !== -1) {
              const jsonStr = buffer.substring(0, newlineIndex).trim();
              if (jsonStr) {
                try {
                  reserveData = JSON.parse(jsonStr) as ReserveResponse;
                  jsonParsed = true;
                  console.log(
                    "[callReserve] Parsed reserve data:",
                    reserveData
                  );
                  if (resolveData) {
                    resolveData(reserveData);
                  }
                } catch (e) {
                  console.error("[callReserve] Failed to parse JSON:", e);
                  if (rejectData) {
                    rejectData(
                      new Error("Failed to parse reserve response JSON")
                    );
                  }
                  break;
                }
              }
            }
          }
          // Continue reading subsequent chunks to keep connection alive
          // The server sends keepalive data periodically (newlines)
        }
      }
    } catch (error) {
      if (abortController.signal.aborted) {
        console.log("[callReserve] Connection aborted by client");
      } else {
        console.error("[callReserve] Error reading stream:", error);
        if (rejectData && !jsonParsed) {
          rejectData(error instanceof Error ? error : new Error(String(error)));
        }
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // Ignore if already released
      }
    }
  };

  // Start reading immediately - this is critical to keep the connection open
  // The reader must start before we await anything to prevent the browser from closing the connection
  const readerPromise = keepAliveReader();

  // Wait for JSON to be parsed (with timeout)
  try {
    reserveData = await Promise.race([
      dataPromise,
      new Promise<ReserveResponse>((_, reject) =>
        setTimeout(
          () => reject(new Error("Timeout waiting for reserve data")),
          5000
        )
      ),
    ]);
  } catch (error) {
    abortController.abort();
    await readerPromise.catch(() => {}); // Wait for reader to finish
    throw error;
  }

  if (!reserveData) {
    abortController.abort();
    await readerPromise.catch(() => {}); // Wait for reader to finish
    throw new Error("Failed to get reserve data");
  }

  // Return immediately with the data, but keep the reader running in background
  // The connection will stay open as long as keepAliveReader continues reading
  // Store the promise reference to prevent garbage collection
  return {
    data: reserveData,
    abortController,
    readerPromise: readerPromise as Promise<void>,
  };
};

/**
 * Build API base URL from host, always using port 8000 for API endpoints
 * The port from /reserve is for reference only, API endpoints are on port 8000
 */
export const buildApiBaseUrl = (host: string): string => {
  if (!host) {
    return API_BASE_URL;
  }
  // Clean the host: remove protocol if present, remove trailing slashes
  let cleanHost = host.trim();
  // Remove http:// or https:// if present
  cleanHost = cleanHost.replace(/^https?:\/\//, "");
  // Remove trailing slashes and any path
  cleanHost = cleanHost.replace(/\/+.*$/, "");
  // Remove any port if present (we'll add our own)
  cleanHost = cleanHost.replace(/:\d+$/, "");

  console.log(
    "[buildApiBaseUrl] Original host:",
    host,
    "Cleaned host:",
    cleanHost
  );

  // Always use port 8000 for API endpoints (WebRTC, pipeline, etc.)
  const protocol = "http";
  const result = `${protocol}://${cleanHost}:8000`;
  console.log("[buildApiBaseUrl] Result:", result);
  return result;
};
