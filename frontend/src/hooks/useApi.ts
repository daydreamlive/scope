/**
 * Unified API hook that automatically routes requests through CloudAdapter
 * when in cloud mode, or uses direct HTTP when in local mode.
 */

import { useCallback } from "react";
import { useCloudContext } from "../lib/cloudContext";
import * as api from "../lib/api";
import type {
  PipelineStatusResponse,
  PipelineLoadRequest,
  PipelineSchemasResponse,
  HardwareInfoResponse,
  LoRAFilesResponse,
  LoRAInstallRequest,
  LoRAInstallResponse,
  AssetsResponse,
  AssetFileInfo,
  WebRTCOfferRequest,
  WebRTCOfferResponse,
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
      if (isCloudMode && adapter) {
        return adapter.api.getPipelineStatus();
      }
      return api.getPipelineStatus();
    }, [adapter, isCloudMode]);

  const loadPipeline = useCallback(
    async (data: PipelineLoadRequest): Promise<{ message: string }> => {
      if (isCloudMode && adapter) {
        return adapter.api.loadPipeline(data);
      }
      return api.loadPipeline(data);
    },
    [adapter, isCloudMode]
  );

  const getPipelineSchemas =
    useCallback(async (): Promise<PipelineSchemasResponse> => {
      if (isCloudMode && adapter) {
        return adapter.api.getPipelineSchemas();
      }
      return api.getPipelineSchemas();
    }, [adapter, isCloudMode]);

  // Model APIs
  const checkModelStatus = useCallback(
    async (pipelineId: string): Promise<ModelStatusResponse> => {
      if (isCloudMode && adapter) {
        return adapter.api.checkModelStatus(pipelineId);
      }
      return api.checkModelStatus(pipelineId);
    },
    [adapter, isCloudMode]
  );

  const downloadPipelineModels = useCallback(
    async (pipelineId: string): Promise<{ message: string }> => {
      if (isCloudMode && adapter) {
        return adapter.api.downloadPipelineModels(pipelineId);
      }
      return api.downloadPipelineModels(pipelineId);
    },
    [adapter, isCloudMode]
  );

  // Hardware APIs
  const getHardwareInfo =
    useCallback(async (): Promise<HardwareInfoResponse> => {
      if (isCloudMode && adapter) {
        return adapter.api.getHardwareInfo();
      }
      return api.getHardwareInfo();
    }, [adapter, isCloudMode]);

  // LoRA APIs
  const listLoRAFiles = useCallback(async (): Promise<LoRAFilesResponse> => {
    if (isCloudMode && adapter) {
      return adapter.api.listLoRAFiles();
    }
    return api.listLoRAFiles();
  }, [adapter, isCloudMode]);

  const installLoRAFile = useCallback(
    async (data: LoRAInstallRequest): Promise<LoRAInstallResponse> => {
      if (isCloudMode && adapter) {
        return adapter.api.installLoRAFile(data);
      }
      return api.installLoRAFile(data);
    },
    [adapter, isCloudMode]
  );

  // Asset APIs
  const listAssets = useCallback(
    async (type?: "image" | "video"): Promise<AssetsResponse> => {
      if (isCloudMode && adapter) {
        return adapter.api.listAssets(type);
      }
      return api.listAssets(type);
    },
    [adapter, isCloudMode]
  );

  const uploadAsset = useCallback(
    async (file: File): Promise<AssetFileInfo> => {
      if (isCloudMode && adapter) {
        return adapter.api.uploadAsset(file);
      }
      return api.uploadAsset(file);
    },
    [adapter, isCloudMode]
  );

  // Logs
  const fetchCurrentLogs = useCallback(async (): Promise<string> => {
    if (isCloudMode && adapter) {
      return adapter.api.fetchCurrentLogs();
    }
    return api.fetchCurrentLogs();
  }, [adapter, isCloudMode]);

  // Recording - note: in cloud mode, we still use direct HTTP for binary download
  const downloadRecording = useCallback(
    async (sessionId: string): Promise<void> => {
      // Always use direct HTTP for binary downloads
      // In cloud mode, this may need the full URL
      return api.downloadRecording(sessionId);
    },
    []
  );

  // WebRTC signaling
  const getIceServers = useCallback(async (): Promise<IceServersResponse> => {
    if (isCloudMode && adapter) {
      return adapter.getIceServers();
    }
    return api.getIceServers();
  }, [adapter, isCloudMode]);

  const sendWebRTCOffer = useCallback(
    async (data: WebRTCOfferRequest): Promise<WebRTCOfferResponse> => {
      if (isCloudMode && adapter) {
        return adapter.sendOffer(
          data.sdp || "",
          data.type || "offer",
          data.initialParameters
        );
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
      if (isCloudMode && adapter) {
        const candidateArray = Array.isArray(candidates)
          ? candidates
          : [candidates];
        for (const candidate of candidateArray) {
          await adapter.sendIceCandidate(sessionId, candidate);
        }
        return;
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
    installLoRAFile,

    // Assets
    listAssets,
    uploadAsset,
    getAssetUrl: api.getAssetUrl, // This is just a URL builder, no API call

    // Logs
    fetchCurrentLogs,

    // Recording
    downloadRecording,

    // WebRTC signaling
    getIceServers,
    sendWebRTCOffer,
    sendIceCandidates,
  };
}
