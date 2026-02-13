/**
 * Unified API hook that automatically routes requests through CloudAdapter
 * when in cloud mode, or uses direct HTTP when in local mode.
 */

import { useCallback } from "react";
import { useCloudContext } from "../lib/directCloudContext";
import * as api from "../lib/api";
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
  ServerInfo,
} from "../lib/api";
import type { IceServersResponse, ModelStatusResponse } from "../types";

/**
 * Hook that provides API functions that work in both local and cloud modes.
 *
 * In cloud mode, all requests go through the CloudAdapter WebSocket.
 * In local mode, requests go directly via HTTP fetch.
 */
export function useApi() {
  const { adapter, isCloudMode, isReady } = useCloudContext();

  // Pipeline APIs
  const getPipelineStatus =
    useCallback(async (): Promise<PipelineStatusResponse> => {
      if (isCloudMode) {
        // In cloud mode, must go through adapter - never fall back to direct HTTP
        if (adapter) {
          return adapter.api.getPipelineStatus();
        }
        // Adapter not ready yet, return default status
        return { status: "not_loaded" };
      }
      return api.getPipelineStatus();
    }, [adapter, isCloudMode]);

  const loadPipeline = useCallback(
    async (data: PipelineLoadRequest): Promise<{ message: string }> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.loadPipeline(data);
        }
        throw new Error("Cloud connection not ready");
      }
      return api.loadPipeline(data);
    },
    [adapter, isCloudMode]
  );

  const getPipelineSchemas =
    useCallback(async (): Promise<PipelineSchemasResponse> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.getPipelineSchemas();
        }
        // Adapter not ready yet, return empty pipelines
        return { pipelines: {} };
      }
      return api.getPipelineSchemas();
    }, [adapter, isCloudMode]);

  // Model APIs
  const checkModelStatus = useCallback(
    async (pipelineId: string): Promise<ModelStatusResponse> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.checkModelStatus(pipelineId);
        }
        throw new Error("Cloud connection not ready");
      }
      return api.checkModelStatus(pipelineId);
    },
    [adapter, isCloudMode]
  );

  const downloadPipelineModels = useCallback(
    async (pipelineId: string): Promise<{ message: string }> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.downloadPipelineModels(pipelineId);
        }
        throw new Error("Cloud connection not ready");
      }
      return api.downloadPipelineModels(pipelineId);
    },
    [adapter, isCloudMode]
  );

  // Hardware APIs
  const getHardwareInfo =
    useCallback(async (): Promise<HardwareInfoResponse> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.getHardwareInfo();
        }
        // Adapter not ready yet, return default hardware info
        return { vram_gb: null, spout_available: false };
      }
      return api.getHardwareInfo();
    }, [adapter, isCloudMode]);

  // LoRA APIs
  const listLoRAFiles = useCallback(async (): Promise<LoRAFilesResponse> => {
    if (isCloudMode) {
      if (adapter) {
        return adapter.api.listLoRAFiles();
      }
      // Adapter not ready yet, return empty list
      return { lora_files: [] };
    }
    return api.listLoRAFiles();
  }, [adapter, isCloudMode]);

  // Asset APIs
  const listAssets = useCallback(
    async (type?: "image" | "video"): Promise<AssetsResponse> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.listAssets(type);
        }
        // Adapter not ready yet, return empty list
        return { assets: [] };
      }
      return api.listAssets(type);
    },
    [adapter, isCloudMode]
  );

  const uploadAsset = useCallback(
    async (file: File): Promise<AssetFileInfo> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.uploadAsset(file);
        }
        throw new Error("Cloud connection not ready");
      }
      return api.uploadAsset(file);
    },
    [adapter, isCloudMode]
  );

  // Logs
  const fetchCurrentLogs = useCallback(async (): Promise<string> => {
    if (isCloudMode) {
      if (adapter) {
        return adapter.api.fetchCurrentLogs();
      }
      throw new Error("Cloud connection not ready");
    }
    return api.fetchCurrentLogs();
  }, [adapter, isCloudMode]);

  // Recording download
  const downloadRecording = useCallback(
    async (sessionId: string): Promise<void> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.downloadRecording(sessionId);
        }
        throw new Error("Cloud connection not ready");
      }
      return api.downloadRecording(sessionId);
    },
    [adapter, isCloudMode]
  );

  // Get asset as data URL (for cloud mode where we can't serve files directly)
  const getAssetDataUrl = useCallback(
    async (assetPath: string): Promise<string> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.api.getAssetDataUrl(assetPath);
        }
        throw new Error("Cloud connection not ready");
      }
      // In local mode, just return the regular URL
      return api.getAssetUrl(assetPath);
    },
    [adapter, isCloudMode]
  );

  // Server info
  const getServerInfo = useCallback(async (): Promise<ServerInfo> => {
    if (isCloudMode) {
      if (adapter) {
        return adapter.api.getServerInfo();
      }
      throw new Error("Cloud connection not ready");
    }
    return api.getServerInfo();
  }, [adapter, isCloudMode]);

  // WebRTC signaling
  const getIceServers = useCallback(async (): Promise<IceServersResponse> => {
    if (isCloudMode) {
      if (adapter) {
        return adapter.getIceServers();
      }
      throw new Error("Cloud connection not ready");
    }
    return api.getIceServers();
  }, [adapter, isCloudMode]);

  const sendWebRTCOffer = useCallback(
    async (data: WebRTCOfferRequest): Promise<WebRTCOfferResponse> => {
      if (isCloudMode) {
        if (adapter) {
          return adapter.sendOffer(
            data.sdp || "",
            data.type || "offer",
            data.initialParameters
          );
        }
        throw new Error("Cloud connection not ready");
      }
      return api.sendWebRTCOffer(data);
    },
    [adapter, isCloudMode]
  );

  const sendIceCandidates = useCallback(
    async (
      sessionId: string,
      candidates: RTCIceCandidate | RTCIceCandidate[]
    ): Promise<void> => {
      if (isCloudMode) {
        if (adapter) {
          const candidateArray = Array.isArray(candidates)
            ? candidates
            : [candidates];
          for (const candidate of candidateArray) {
            await adapter.sendIceCandidate(sessionId, candidate);
          }
          return;
        }
        throw new Error("Cloud connection not ready");
      }
      return api.sendIceCandidates(sessionId, candidates);
    },
    [adapter, isCloudMode]
  );

  return {
    // State
    isCloudMode,
    isReady,

    // Pipeline
    getPipelineStatus,
    loadPipeline,
    getPipelineSchemas,

    // Models
    checkModelStatus,
    downloadPipelineModels,

    // Hardware
    getHardwareInfo,

    // LoRA
    listLoRAFiles,

    // Assets
    listAssets,
    uploadAsset,
    getAssetUrl: api.getAssetUrl, // URL builder for local mode
    getAssetDataUrl, // Async fetch for cloud mode (returns data URL)

    // Logs
    fetchCurrentLogs,

    // Recording
    downloadRecording,

    // Server info
    getServerInfo,

    // WebRTC signaling
    getIceServers,
    sendWebRTCOffer,
    sendIceCandidates,
  };
}
