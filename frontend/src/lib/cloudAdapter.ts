/**
 * cloud WebSocket Adapter for Scope
 *
 * This adapter routes all API calls and WebRTC signaling through a single
 * WebSocket connection to the cloud endpoint, preventing cloud from spawning
 * new runner instances for each request.
 *
 * Usage:
 *   const adapter = new CloudAdapter("wss://your-cloud-endpoint/ws", "your-api-key");
 *   await adapter.connect();
 *
 *   // Use like regular API
 *   const status = await adapter.api.getPipelineStatus();
 *
 *   // WebRTC signaling
 *   const iceServers = await adapter.getIceServers();
 *   const answer = await adapter.sendOffer(sdp, type, initialParams);
 *   await adapter.sendIceCandidate(sessionId, candidate);
 *
 * Authentication:
 *   The API key is passed as a query parameter (fal_jwt_token) since
 *   browser WebSocket API doesn't support custom headers.
 */

import type { IceServersResponse, ModelStatusResponse } from "../types";
import type {
  WebRTCOfferRequest,
  WebRTCOfferResponse,
  PipelineStatusResponse,
  PipelineLoadRequest,
  PipelineSchemasResponse,
  HardwareInfoResponse,
  LoRAFilesResponse,
  LoRAInstallRequest,
  LoRAInstallResponse,
  AssetsResponse,
  AssetFileInfo,
} from "./api";

type MessageHandler = (response: ApiResponse) => void;

interface ApiResponse {
  type: string;
  request_id?: string;
  status?: number;
  data?: unknown;
  error?: string;
  // WebRTC specific
  sdp?: string;
  sdp_type?: string;
  sessionId?: string;
  candidate?: RTCIceCandidateInit | null;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}

export class CloudAdapter {
  private ws: WebSocket | null = null;
  private wsUrl: string;
  private apiKey: string | null = null;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private requestCounter = 0;
  private isReady = false;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private messageHandlers: Set<MessageHandler> = new Set();

  // Current WebRTC session ID (set after offer/answer exchange)
  private currentSessionId: string | null = null;

  /**
   * Create a CloudAdapter instance.
   * @param wsUrl - WebSocket URL for the cloud endpoint
   * @param apiKey - Optional cloud API key for authentication
   */
  constructor(wsUrl: string, apiKey?: string) {
    this.wsUrl = wsUrl;
    this.apiKey = apiKey || null;
  }

  /**
   * Connect to the cloud WebSocket endpoint
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.readyPromise = new Promise(resolve => {
      this.readyResolve = resolve;
    });

    return new Promise((resolve, reject) => {
      try {
        // Build URL with auth token as query parameter if provided
        // (WebSocket API doesn't support custom headers in browsers)
        let url = this.wsUrl;
        if (this.apiKey) {
          const separator = url.includes("?") ? "&" : "?";
          url = `${url}${separator}fal_jwt_token=${encodeURIComponent(this.apiKey)}`;
        }

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log("[CloudAdapter] WebSocket connected");
          this.reconnectAttempts = 0;
        };

        this.ws.onmessage = event => {
          try {
            const message = JSON.parse(event.data) as ApiResponse;
            this.handleMessage(message);

            // Check for ready message
            if (message.type === "ready") {
              this.isReady = true;
              this.readyResolve?.();
              resolve();
            }
          } catch (error) {
            console.error("[CloudAdapter] Failed to parse message:", error);
          }
        };

        this.ws.onerror = error => {
          console.error("[CloudAdapter] WebSocket error:", error);
          reject(error);
        };

        this.ws.onclose = event => {
          console.log(
            "[CloudAdapter] WebSocket closed:",
            event.code,
            event.reason
          );
          this.isReady = false;
          this.ws = null;

          // Reject all pending requests
          for (const [requestId, pending] of this.pendingRequests) {
            clearTimeout(pending.timeout);
            pending.reject(new Error("WebSocket connection closed"));
            this.pendingRequests.delete(requestId);
          }

          // Attempt reconnect if not intentional close
          if (
            event.code !== 1000 &&
            this.reconnectAttempts < this.maxReconnectAttempts
          ) {
            this.reconnectAttempts++;
            const delay = Math.min(
              1000 * Math.pow(2, this.reconnectAttempts),
              30000
            );
            console.log(`[CloudAdapter] Reconnecting in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Wait for the adapter to be ready
   */
  async waitForReady(): Promise<void> {
    if (this.isReady) return;
    if (this.readyPromise) {
      await this.readyPromise;
    }
  }

  /**
   * Disconnect from the WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
    this.isReady = false;
  }

  /**
   * Add a message handler for handling server-pushed messages
   */
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  private handleMessage(message: ApiResponse): void {
    // Handle response to pending request
    if (message.request_id && this.pendingRequests.has(message.request_id)) {
      const pending = this.pendingRequests.get(message.request_id)!;
      clearTimeout(pending.timeout);
      this.pendingRequests.delete(message.request_id);

      if (
        message.type === "error" ||
        (message.status && message.status >= 400)
      ) {
        pending.reject(
          new Error(
            message.error || `Request failed with status ${message.status}`
          )
        );
      } else {
        pending.resolve(message);
      }
      return;
    }

    // Handle WebRTC signaling responses (no request_id)
    if (
      message.type === "answer" ||
      message.type === "ice_servers" ||
      message.type === "icecandidate_ack"
    ) {
      // These are handled by specific pending requests
      return;
    }

    // Notify all message handlers for server-pushed messages
    for (const handler of this.messageHandlers) {
      try {
        handler(message);
      } catch (error) {
        console.error("[CloudAdapter] Message handler error:", error);
      }
    }
  }

  private generateRequestId(): string {
    return `req_${++this.requestCounter}_${Date.now()}`;
  }

  private async sendAndWait<T>(
    message: Record<string, unknown>,
    timeoutMs = 30000
  ): Promise<T> {
    await this.waitForReady();

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }

    const requestId = this.generateRequestId();
    const messageWithId = { ...message, request_id: requestId };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Request timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timeout,
      });

      this.ws!.send(JSON.stringify(messageWithId));
    });
  }

  // ==================== WebRTC Signaling ====================

  /**
   * Get ICE servers from the backend
   */
  async getIceServers(): Promise<IceServersResponse> {
    const response = await this.sendAndWait<ApiResponse>({
      type: "get_ice_servers",
    });
    return response.data as IceServersResponse;
  }

  /**
   * Send WebRTC offer and get answer
   */
  async sendOffer(
    sdp: string,
    sdpType: string,
    initialParameters?: WebRTCOfferRequest["initialParameters"]
  ): Promise<WebRTCOfferResponse> {
    const response = await this.sendAndWait<ApiResponse>({
      type: "offer",
      sdp,
      sdp_type: sdpType,
      initialParameters,
    });

    if (response.sessionId) {
      this.currentSessionId = response.sessionId;
    }

    return {
      sdp: response.sdp!,
      type: response.sdp_type!,
      sessionId: response.sessionId!,
    };
  }

  /**
   * Send ICE candidate
   */
  async sendIceCandidate(
    sessionId: string | null,
    candidate: RTCIceCandidate | null
  ): Promise<void> {
    await this.sendAndWait<ApiResponse>({
      type: "icecandidate",
      sessionId: sessionId || this.currentSessionId,
      candidate: candidate
        ? {
            candidate: candidate.candidate,
            sdpMid: candidate.sdpMid,
            sdpMLineIndex: candidate.sdpMLineIndex,
          }
        : null,
    });
  }

  /**
   * Send multiple ICE candidates
   */
  async sendIceCandidates(
    sessionId: string,
    candidates: RTCIceCandidate[]
  ): Promise<void> {
    for (const candidate of candidates) {
      await this.sendIceCandidate(sessionId, candidate);
    }
  }

  // ==================== API Proxy ====================

  /**
   * Make an API request through the WebSocket
   */
  private async apiRequest<T>(
    method: "GET" | "POST" | "PATCH" | "DELETE",
    path: string,
    body?: unknown,
    timeoutMs?: number
  ): Promise<T> {
    const response = await this.sendAndWait<ApiResponse>(
      {
        type: "api",
        method,
        path,
        body,
      },
      timeoutMs
    );

    if (response.status && response.status >= 400) {
      throw new Error(
        response.error || `API request failed with status ${response.status}`
      );
    }

    return response.data as T;
  }

  // API methods matching the original api.ts interface
  api = {
    getPipelineStatus: (): Promise<PipelineStatusResponse> =>
      this.apiRequest("GET", "/api/v1/pipeline/status"),

    loadPipeline: (data: PipelineLoadRequest): Promise<{ message: string }> =>
      this.apiRequest("POST", "/api/v1/pipeline/load", data),

    getPipelineSchemas: (): Promise<PipelineSchemasResponse> =>
      this.apiRequest("GET", "/api/v1/pipelines/schemas"),

    checkModelStatus: (pipelineId: string): Promise<ModelStatusResponse> =>
      this.apiRequest("GET", `/api/v1/models/status?pipeline_id=${pipelineId}`),

    downloadPipelineModels: (
      pipelineId: string
    ): Promise<{ message: string }> =>
      this.apiRequest("POST", "/api/v1/models/download", {
        pipeline_id: pipelineId,
      }),

    getHardwareInfo: (): Promise<HardwareInfoResponse> =>
      this.apiRequest("GET", "/api/v1/hardware/info"),

    listLoRAFiles: (): Promise<LoRAFilesResponse> =>
      this.apiRequest("GET", "/api/v1/loras"),

    installLoRAFile: (data: LoRAInstallRequest): Promise<LoRAInstallResponse> =>
      this.apiRequest("POST", "/api/v1/loras", data, 300000),

    listAssets: (type?: "image" | "video"): Promise<AssetsResponse> =>
      this.apiRequest(
        "GET",
        type ? `/api/v1/assets?type=${type}` : "/api/v1/assets"
      ),

    uploadAsset: async (file: File): Promise<AssetFileInfo> => {
      // For file uploads, we need to convert to base64 and send through WebSocket
      // This is a limitation of the WebSocket approach
      const arrayBuffer = await file.arrayBuffer();
      const base64 = btoa(
        new Uint8Array(arrayBuffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ""
        )
      );

      return this.apiRequest(
        "POST",
        `/api/v1/assets?filename=${encodeURIComponent(file.name)}`,
        {
          _base64_content: base64,
          _content_type: file.type,
        }
      );
    },

    fetchCurrentLogs: (): Promise<string> =>
      this.apiRequest("GET", "/api/v1/logs/current"),

    // Note: downloadRecording needs special handling for binary data
    // For now, it will return the URL to download from
    getRecordingUrl: (sessionId: string): string =>
      `/api/v1/recordings/${sessionId}`,
  };
}

// ==================== React Hook ====================

import { useState, useEffect, useRef, useCallback } from "react";

/**
 * React hook for using the CloudAdapter
 */
export function useCloudAdapter(wsUrl: string | null, apiKey?: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const adapterRef = useRef<CloudAdapter | null>(null);

  useEffect(() => {
    if (!wsUrl) {
      adapterRef.current = null;
      setIsConnected(false);
      setIsReady(false);
      return;
    }

    const adapter = new CloudAdapter(wsUrl, apiKey);
    adapterRef.current = adapter;

    adapter
      .connect()
      .then(() => {
        setIsConnected(true);
        setIsReady(true);
        setError(null);
      })
      .catch(err => {
        setError(err);
        setIsConnected(false);
        setIsReady(false);
      });

    return () => {
      adapter.disconnect();
    };
  }, [wsUrl, apiKey]);

  const getAdapter = useCallback(() => adapterRef.current, []);

  return {
    adapter: adapterRef.current,
    getAdapter,
    isConnected,
    isReady,
    error,
  };
}

// ==================== Global Instance ====================

let globalAdapter: CloudAdapter | null = null;

/**
 * Initialize the global CloudAdapter instance
 * Call this once at app startup if using cloud deployment
 */
export function initCloudAdapter(wsUrl: string, apiKey?: string): CloudAdapter {
  if (globalAdapter) {
    globalAdapter.disconnect();
  }
  globalAdapter = new CloudAdapter(wsUrl, apiKey);
  return globalAdapter;
}

/**
 * Get the global CloudAdapter instance
 */
export function getCloudAdapter(): CloudAdapter | null {
  return globalAdapter;
}

/**
 * Check if we're running in cloud mode (adapter is initialized)
 */
export function isCloudMode(): boolean {
  return globalAdapter !== null && globalAdapter !== undefined;
}
