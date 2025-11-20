import type { LoRAConfig } from "../types";
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

export interface LongLiveLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
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
  retries: number = 3,
  retryDelay: number = 1000
): Promise<RTCSessionDescriptionInit> => {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // Create an AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch("/api/v1/webrtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();

        // Check if it's a Cloudflare timeout error (524) or bad gateway (502)
        if (response.status === 524 || response.status === 502) {
          // If we have retries left, wait and retry
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
            continue;
          }
          // Otherwise throw with a more helpful message
          const errorType = response.status === 524 ? "timeout (524)" : "bad gateway (502)";
          throw new Error(
            `WebRTC offer ${errorType}: Cloudflare error. The pipeline may still be loading.`
          );
        }

        throw new Error(
          `WebRTC offer failed: ${response.status} ${response.statusText}: ${errorText}`
        );
      }

      const result = await response.json();
      return result;
    } catch (error) {
      // Handle AbortError from timeout
      if (error instanceof Error && error.name === "AbortError") {
        if (attempt < retries) {
          await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
          continue;
        }
        throw new Error(
          `WebRTC offer request timed out. The pipeline may still be loading.`
        );
      }

      // Check if error message contains 524/502 (Cloudflare errors)
      const isCloudflareTimeout = error instanceof Error &&
        (error.message.includes("524") || error.message.includes("502") ||
         error.message.includes("timeout") || error.message.includes("timed out") ||
         error.message.includes("bad gateway"));

      // If it's the last attempt or not a retryable error, throw
      if (attempt === retries || !isCloudflareTimeout) {
        throw error;
      }

      // Wait before retrying with exponential backoff
      await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
    }
  }

  // This should never be reached, but TypeScript needs it
  throw new Error("Failed to send WebRTC offer after retries");
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

export const getPipelineStatus = async (
  retries: number = 3,
  retryDelay: number = 1000
): Promise<PipelineStatusResponse> => {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // Create an AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch("/api/v1/pipeline/status", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();

        // Check if it's a Cloudflare timeout error (524) or bad gateway (502)
        if (response.status === 524 || response.status === 502) {
          // If we have retries left, wait and retry
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
            continue;
          }
          // Otherwise throw with a more helpful message
          const errorType = response.status === 524 ? "timeout (524)" : "bad gateway (502)";
          throw new Error(
            `Pipeline status ${errorType}: Cloudflare error. The pipeline may still be loading.`
          );
        }

        throw new Error(
          `Pipeline status failed: ${response.status} ${response.statusText}: ${errorText}`
        );
      }

      const result = await response.json();
      return result;
    } catch (error) {
      // Handle AbortError from timeout
      if (error instanceof Error && error.name === "AbortError") {
        if (attempt < retries) {
          await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
          continue;
        }
        throw new Error(
          `Pipeline status request timed out. The pipeline may still be loading.`
        );
      }

      // Check if error message contains 524/502 (Cloudflare errors)
      const isCloudflareTimeout = error instanceof Error &&
        (error.message.includes("524") || error.message.includes("502") ||
         error.message.includes("timeout") || error.message.includes("timed out") ||
         error.message.includes("bad gateway"));

      // If it's the last attempt or not a retryable error, throw
      if (attempt === retries || !isCloudflareTimeout) {
        throw error;
      }

      // Wait before retrying with exponential backoff
      await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
    }
  }

  // This should never be reached, but TypeScript needs it
  throw new Error("Failed to get pipeline status after retries");
};

export const checkModelStatus = async (
  pipelineId: string,
  retries: number = 3,
  retryDelay: number = 1000
): Promise<{ downloaded: boolean }> => {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // Create an AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(
        `/api/v1/models/status?pipeline_id=${pipelineId}`,
        {
          method: "GET",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
        }
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();

        // Check if it's a Cloudflare timeout error (524) or bad gateway (502)
        if (response.status === 524 || response.status === 502) {
          // If we have retries left, wait and retry
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
            continue;
          }
          // Otherwise throw with a more helpful message
          const errorType = response.status === 524 ? "timeout (524)" : "bad gateway (502)";
          throw new Error(
            `Model status check ${errorType}: Cloudflare error. The pipeline may still be loading.`
          );
        }

        throw new Error(
          `Model status check failed: ${response.status} ${response.statusText}: ${errorText}`
        );
      }

      const result = await response.json();
      return result;
    } catch (error) {
      // Handle AbortError from timeout
      if (error instanceof Error && error.name === "AbortError") {
        if (attempt < retries) {
          await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
          continue;
        }
        throw new Error(
          `Model status check request timed out. The pipeline may still be loading.`
        );
      }

      // Check if error message contains 524/502 (Cloudflare errors)
      const isCloudflareTimeout = error instanceof Error &&
        (error.message.includes("524") || error.message.includes("502") ||
         error.message.includes("timeout") || error.message.includes("timed out") ||
         error.message.includes("bad gateway"));

      // If it's the last attempt or not a retryable error, throw
      if (attempt === retries || !isCloudflareTimeout) {
        throw error;
      }

      // Wait before retrying with exponential backoff
      await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
    }
  }

  // This should never be reached, but TypeScript needs it
  throw new Error("Failed to check model status after retries");
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
