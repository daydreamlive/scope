/**
 * Unified API hook that automatically routes requests through CloudAdapter
 * when in cloud mode, or uses direct HTTP when in local mode.
 */

import { useCallback } from "react";
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
  // Pipeline APIs
  const getPipelineStatus =
    useCallback(async (): Promise<PipelineStatusResponse> => {
      return api.getPipelineStatus();
    }, []);

  const loadPipeline = useCallback(
    async (data: PipelineLoadRequest): Promise<{ message: string }> => {
      return api.loadPipeline(data);
    },
    []
  );

  const getPipelineSchemas =
    useCallback(async (): Promise<PipelineSchemasResponse> => {
      return api.getPipelineSchemas();
    }, []);

  // Model APIs
  const checkModelStatus = useCallback(
    async (pipelineId: string): Promise<ModelStatusResponse> => {
      return api.checkModelStatus(pipelineId);
    },
    []
  );

  const downloadPipelineModels = useCallback(
    async (pipelineId: string): Promise<{ message: string }> => {
      return api.downloadPipelineModels(pipelineId);
    },
    []
  );

  // Hardware APIs
  const getHardwareInfo =
    useCallback(async (): Promise<HardwareInfoResponse> => {
      return api.getHardwareInfo();
    }, []);

  // LoRA APIs
  const listLoRAFiles = useCallback(async (): Promise<LoRAFilesResponse> => {
    return api.listLoRAFiles();
  }, []);

  const installLoRAFile = useCallback(
    async (data: LoRAInstallRequest): Promise<LoRAInstallResponse> => {
      return api.installLoRAFile(data);
    },
    []
  );

  // Asset APIs
  const listAssets = useCallback(
    async (type?: "image" | "video"): Promise<AssetsResponse> => {
      return api.listAssets(type);
    },
    []
  );

  const uploadAsset = useCallback(
    async (file: File): Promise<AssetFileInfo> => {
      return api.uploadAsset(file);
    },
    []
  );

  // Logs
  const fetchCurrentLogs = useCallback(async (): Promise<string> => {
    return api.fetchCurrentLogs();
  }, []);

  // Recording - note: in cloud mode, we still use direct HTTP for binary download
  const downloadRecording = useCallback(
    async (sessionId: string, nodeId?: string): Promise<void> => {
      return api.downloadRecording(sessionId, nodeId);
    },
    []
  );

  const startRecording = useCallback(
    async (sessionId: string, nodeId?: string): Promise<{ status: string }> => {
      return api.startRecording(sessionId, nodeId);
    },
    []
  );

  const stopRecording = useCallback(
    async (sessionId: string, nodeId?: string): Promise<{ status: string }> => {
      return api.stopRecording(sessionId, nodeId);
    },
    []
  );

  // WebRTC signaling
  const getIceServers = useCallback(async (): Promise<IceServersResponse> => {
    return api.getIceServers();
  }, []);

  const sendWebRTCOffer = useCallback(
    async (data: WebRTCOfferRequest): Promise<WebRTCOfferResponse> => {
      return api.sendWebRTCOffer(data);
    },
    []
  );

  const sendIceCandidates = useCallback(
    async (
      sessionId: string,
      candidates: RTCIceCandidate | RTCIceCandidate[]
    ): Promise<void> => {
      return api.sendIceCandidates(sessionId, candidates);
    },
    []
  );

  return {
    // State
    isCloudMode: false,
    isReady: true,

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
    startRecording,
    stopRecording,

    // WebRTC signaling
    getIceServers,
    sendWebRTCOffer,
    sendIceCandidates,
  };
}
