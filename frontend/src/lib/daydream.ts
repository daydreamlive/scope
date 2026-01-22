import { getDaydreamAPIKey } from "./auth";

const DAYDREAM_API_BASE =
  (import.meta as any).env?.VITE_DAYDREAM_API_BASE || "https://api.daydream.monster";

/**
 * Prompt can be a single string or weighted list of (prompt, weight) tuples
 */
export type PromptInput =
  | string
  | Array<[string, number]>;

/**
 * Seed can be a single integer or weighted list of (seed, weight) tuples
 */
export type SeedInput =
  | number
  | Array<[number, number]>;

/**
 * ControlNet configuration for guided generation
 */
export interface ControlNetConfig {
  /** ControlNet model identifier. Must be unique and compatible with base model */
  model_id: string;
  /** Strength of ControlNet's influence (0.0-1.0) */
  conditioning_scale: number;
  /** Preprocessor to apply to input frames */
  preprocessor:
    | "pose_tensorrt"
    | "soft_edge"
    | "canny"
    | "depth_tensorrt"
    | "passthrough"
    | "feedback";
  /** Additional parameters for the preprocessor */
  preprocessor_params?: Record<string, unknown>;
  /** Whether this ControlNet is active */
  enabled: boolean;
  /** Fraction of denoising process (0.0-1.0) when guidance begins */
  control_guidance_start?: number;
  /** Fraction of denoising process (0.0-1.0) when guidance ends */
  control_guidance_end?: number;
}

/**
 * IP Adapter configuration for style conditioning
 */
export interface IPAdapterConfig {
  /** Type of IP adapter: 'faceid' for SDXL-faceid models, 'regular' for others */
  type: "regular" | "faceid";
  /** Whether IP adapter is enabled */
  enabled: boolean;
  /** Strength of IP adapter style conditioning */
  scale: number;
  /** Weight interpolation method for IP adapter */
  weight_type:
    | "linear"
    | "ease in"
    | "ease out"
    | "ease in-out"
    | "reverse in-out"
    | "weak input"
    | "weak output"
    | "weak middle"
    | "strong middle"
    | "style transfer"
    | "composition"
    | "strong style transfer"
    | "style and composition"
    | "style transfer precise"
    | "composition precise";
}

/**
 * Pipeline parameters for stream updates. Structure depends on model_id.
 * Based on Daydream API documentation for SDTurbo, SDXL, and SD1.5 models.
 */
export interface StreamParams {
  /** Model identifier */
  model_id?: string;
  /** Text prompt describing the desired image */
  prompt?: PromptInput;
  /** Method for interpolating between multiple prompts */
  prompt_interpolation_method?: "linear" | "slerp";
  /** Whether to normalize prompt weights to sum to 1.0 */
  normalize_prompt_weights?: boolean;
  /** Whether to normalize seed weights to sum to 1.0 */
  normalize_seed_weights?: boolean;
  /** Text describing what to avoid in the generated image */
  negative_prompt?: string;
  /** Strength of prompt adherence. Higher values make model follow prompt more strictly */
  guidance_scale?: number;
  /** Delta sets per-frame denoising progress */
  delta?: number;
  /** Builds the full denoising schedule. Typical range 10-200, default 50 */
  num_inference_steps?: number;
  /** Ordered list of step indices from num_inference_steps schedule to execute per frame (1-4 elements) */
  t_index_list?: number[];
  /** Whether to use safety checker for content filtering */
  use_safety_checker?: boolean;
  /** Output image width in pixels. Must be divisible by 64 and between 384-1024 */
  width?: number;
  /** Output image height in pixels. Must be divisible by 64 and between 384-1024 */
  height?: number;
  /** Dictionary mapping LoRA model paths to their weights */
  lora_dict?: Record<string, number> | null;
  /** Whether to use Latent Consistency Model LoRA for faster inference */
  use_lcm_lora?: boolean;
  /** Identifier for the LCM LoRA model to use */
  lcm_lora_id?: string;
  /** Acceleration method for inference */
  acceleration?: "none" | "xformers" | "tensorrt";
  /** Whether to process multiple denoising steps in a single batch */
  use_denoising_batch?: boolean;
  /** Whether to add noise to input frames before processing */
  do_add_noise?: boolean;
  /** Random seed for generation */
  seed?: SeedInput;
  /** Method for interpolating between multiple seeds */
  seed_interpolation_method?: "linear" | "slerp";
  /** Whether to skip frames that are too similar to previous output */
  enable_similar_image_filter?: boolean;
  /** Similarity threshold for the image filter (0.0-1.0) */
  similar_image_filter_threshold?: number;
  /** Maximum number of consecutive frames that can be skipped */
  similar_image_filter_max_skip_frame?: number;
  /** List of ControlNet configurations for guided generation */
  controlnets?: ControlNetConfig[];
  /** IP adapter configuration (available for SDXL, SDXL-faceid, SD1.5) */
  ip_adapter?: IPAdapterConfig;
  /** HTTPS URL or base64-encoded data URI of style image for IP adapter */
  ip_adapter_style_image_url?: string;
}

/**
 * Request payload for creating a new stream via the Daydream API.
 *
 * @see https://api.daydream.live/v1/streams
 */
interface CreateStreamRequest {
  /** Pipeline type to use for the stream. Currently only "streamdiffusion" is supported. */
  pipeline: string;
  /** Pipeline configuration parameters. The structure depends on the model_id selected. */
  params?: StreamParams;
  /** Human-readable name for the stream */
  name?: string;
  /** Custom RTMP URL for stream output destination */
  output_rtmp_url?: string;
}

export interface CreateStreamResponse {
  pipeline: string;
  params: Record<string, unknown>;
  id: string;
  stream_key: string;
  created_at: string;
  output_playback_id: string;
  name?: string;
  author?: string;
  from_playground?: boolean;
  gateway_host?: string;
  is_smoke_test?: boolean;
  whip_url?: string;
  // other fields may exist
}

/**
 * Request payload for updating an existing stream via the Daydream API.
 *
 * @see https://api.daydream.live/v1/streams/{id}
 */
export interface UpdateStreamRequest {
  /** Updated pipeline parameters for the stream */
  params?: StreamParams;
  /** Updated prompt to apply to the stream processing */
  prompt?: string;
}

interface UpdateStreamResponse {
  pipeline: string;
  params: Record<string, unknown>;
  id: string;
  stream_key: string;
  created_at: string;
  output_playback_id: string;
  name?: string;
  author?: string;
  from_playground?: boolean;
  gateway_host?: string;
  is_smoke_test?: boolean;
  whip_url?: string;
  // other fields may exist
}

export interface DaydreamWebRTCOfferRequest {
  sdp?: string;
  type?: string;
  initialParameters?: Record<string, unknown>;
}

export async function createCloudStream(req: CreateStreamRequest) {
  const apiKey = getDaydreamAPIKey();
  if (!apiKey) {
    throw new Error("Not authenticated. Please sign in to use Daydream API.");
  }

  const response = await fetch(`${DAYDREAM_API_BASE}/v1/streams`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Daydream create stream failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = (await response.json()) as CreateStreamResponse;
  return result;
}

/**
 * Updates pipeline parameters for an existing video processing stream.
 *
 * @param streamId - ID of the stream to update
 * @param req - Update request containing params and/or prompt
 * @returns Updated stream information
 * @see https://api.daydream.live/v1/streams/{id}
 */
export async function updateCloudStream(
  streamId: string,
  req: UpdateStreamRequest
): Promise<UpdateStreamResponse> {
  const apiKey = getDaydreamAPIKey();
  if (!apiKey) {
    throw new Error("Not authenticated. Please sign in to use Daydream API.");
  }

  const response = await fetch(
    `${DAYDREAM_API_BASE}/v1/streams/${encodeURIComponent(streamId)}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(req),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Daydream update stream failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = (await response.json()) as UpdateStreamResponse;
  return result;
}

export async function sendDaydreamWebRTCOffer(
  streamId: string,
  data: DaydreamWebRTCOfferRequest
): Promise<RTCSessionDescriptionInit> {
  const apiKey = getDaydreamAPIKey();
  if (!apiKey) {
    throw new Error("Not authenticated. Please sign in to use Daydream API.");
  }

  const response = await fetch(
    `${DAYDREAM_API_BASE}/v1/streams/${encodeURIComponent(streamId)}/webrtc/offer`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(data),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Daydream WebRTC offer failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
}
