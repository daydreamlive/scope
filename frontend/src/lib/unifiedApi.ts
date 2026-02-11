import type { CloudAdapter } from "./cloudAdapter";
import * as api from "./api";
import type {
  PipelineStatusResponse,
  PipelineLoadRequest,
  PipelineSchemasResponse,
  HardwareInfoResponse,
  LoRAFilesResponse,
  AssetsResponse,
  AssetFileInfo,
  WebRTCOfferRequest,
  WebRTCOfferResponse,
} from "./api";
import type { IceServersResponse, ModelStatusResponse } from "../types";

export interface ApiClient {
  // Pipeline
  getPipelineStatus: () => Promise<PipelineStatusResponse>;
  loadPipeline: (data: PipelineLoadRequest) => Promise<{ message: string }>;
  getPipelineSchemas: () => Promise<PipelineSchemasResponse>;

  // Models
  checkModelStatus: (pipelineId: string) => Promise<ModelStatusResponse>;
  downloadPipelineModels: (pipelineId: string) => Promise<{ message: string }>;

  // Hardware
  getHardwareInfo: () => Promise<HardwareInfoResponse>;

  // LoRA
  listLoRAFiles: () => Promise<LoRAFilesResponse>;

  // Assets
  listAssets: (type?: "image" | "video") => Promise<AssetsResponse>;
  uploadAsset: (file: File) => Promise<AssetFileInfo>;
  getAssetUrl: (assetPath: string) => string;

  // Logs
  fetchCurrentLogs: () => Promise<string>;

  // Recording (always direct HTTP for binary download)
  downloadRecording: (sessionId: string) => Promise<void>;

  // WebRTC signaling
  getIceServers: () => Promise<IceServersResponse>;
  sendWebRTCOffer: (data: WebRTCOfferRequest) => Promise<WebRTCOfferResponse>;
  sendIceCandidates: (
    sessionId: string,
    candidates: RTCIceCandidate | RTCIceCandidate[]
  ) => Promise<void>;
}

/**
 * Creates an API client that routes through CloudAdapter when in cloud mode.
 */
export function createApiClient(
  adapter: CloudAdapter | null,
  isCloudMode: boolean
): ApiClient {
  const route = <T>(
    localFn: () => Promise<T>,
    cloudFn: () => Promise<T>
  ): (() => Promise<T>) => (isCloudMode && adapter ? cloudFn : localFn);

  const routeWith = <A extends unknown[], T>(
    localFn: (...args: A) => Promise<T>,
    cloudFn: (...args: A) => Promise<T>
  ): ((...args: A) => Promise<T>) =>
    isCloudMode && adapter ? cloudFn : localFn;

  return {
    // Pipeline
    getPipelineStatus: route(api.getPipelineStatus, () =>
      adapter!.api.getPipelineStatus()
    ),
    loadPipeline: routeWith(api.loadPipeline, data =>
      adapter!.api.loadPipeline(data)
    ),
    getPipelineSchemas: route(api.getPipelineSchemas, () =>
      adapter!.api.getPipelineSchemas()
    ),

    // Models
    checkModelStatus: routeWith(api.checkModelStatus, pipelineId =>
      adapter!.api.checkModelStatus(pipelineId)
    ),
    downloadPipelineModels: routeWith(api.downloadPipelineModels, pipelineId =>
      adapter!.api.downloadPipelineModels(pipelineId)
    ),

    // Hardware
    getHardwareInfo: route(api.getHardwareInfo, () =>
      adapter!.api.getHardwareInfo()
    ),

    // LoRA
    listLoRAFiles: route(api.listLoRAFiles, () => adapter!.api.listLoRAFiles()),

    // Assets
    listAssets: routeWith(api.listAssets, (type?) =>
      adapter!.api.listAssets(type)
    ),
    uploadAsset: routeWith(api.uploadAsset, file =>
      adapter!.api.uploadAsset(file)
    ),
    getAssetUrl: api.getAssetUrl,

    // Logs
    fetchCurrentLogs: route(api.fetchCurrentLogs, () =>
      adapter!.api.fetchCurrentLogs()
    ),

    // Recording - always direct HTTP for binary download
    downloadRecording: api.downloadRecording,

    // WebRTC signaling (special cloud handling)
    getIceServers: route(api.getIceServers, () => adapter!.getIceServers()),
    sendWebRTCOffer: routeWith(api.sendWebRTCOffer, data =>
      adapter!.sendOffer(
        data.sdp || "",
        data.type || "offer",
        data.initialParameters
      )
    ),
    sendIceCandidates: routeWith(
      api.sendIceCandidates,
      async (sessionId, candidates) => {
        const candidateArray = Array.isArray(candidates)
          ? candidates
          : [candidates];
        for (const candidate of candidateArray) {
          await adapter!.sendIceCandidate(sessionId, candidate);
        }
      }
    ),
  };
}
