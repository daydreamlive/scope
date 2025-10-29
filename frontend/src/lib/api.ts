export interface PromptItem {
  text: string;
  weight: number;
}

export interface WebRTCOfferRequest {
  sdp?: string;
  type?: string;
  initialParameters?: {
    prompts?: string[] | PromptItem[];
    prompt_interpolation_method?: "linear" | "slerp";
    denoising_step_list?: number[];
    noise_scale?: number;
    noise_controller?: boolean;
    manage_cache?: boolean;
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
}

export interface KreaRealtimeVideoLoadParams extends PipelineLoadParams {
  height?: number;
  width?: number;
  seed?: number;
  quantization?: "fp8_e4m3fn" | null;
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
  error?: string;
}

export const sendWebRTCOffer = async (
  data: WebRTCOfferRequest
): Promise<RTCSessionDescriptionInit> => {
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
