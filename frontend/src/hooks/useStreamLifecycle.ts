/**
 * Stream lifecycle handlers.
 *
 * Manages starting/stopping streams (including cloud connection wait,
 * model download checks, pipeline loading) and saving recordings.
 */

import { useCallback } from "react";
import { useAppStore } from "../stores";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useApi } from "./useApi";
import { adjustResolutionForPipeline } from "../lib/utils";
import { toast } from "sonner";
import type {
  InputMode,
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
} from "../types";
import type { VideoSourceMode } from "./useVideoSource";

function buildLoRAParams(
  loras?: LoRAConfig[],
  strategy?: LoraMergeStrategy
): {
  loras?: { path: string; scale: number; merge_mode?: string }[];
  lora_merge_mode: string;
} {
  return {
    loras: loras?.map(({ path, scale, mergeMode }) => ({
      path,
      scale,
      ...(mergeMode && { merge_mode: mergeMode }),
    })),
    lora_merge_mode: strategy ?? "permanent_merge",
  };
}

function getVaceParams(
  refImages?: string[],
  vaceContextScale?: number
):
  | { vace_ref_images: string[]; vace_context_scale: number }
  | Record<string, never> {
  if (refImages && refImages.length > 0) {
    return {
      vace_ref_images: refImages,
      vace_context_scale: vaceContextScale ?? 1.0,
    };
  }
  return {};
}

async function waitForCloudConnection(): Promise<boolean> {
  const maxWaitMs = 180_000;
  const pollIntervalMs = 2000;
  const start = Date.now();

  while (Date.now() - start < maxWaitMs) {
    try {
      const response = await fetch("/api/v1/cloud/status");
      if (response.ok) {
        const data = await response.json();
        if (data.connected) return true;
        if (!data.connecting) {
          console.error("Cloud connection failed:", data.error);
          return false;
        }
      }
    } catch (e) {
      console.error("Error polling cloud status:", e);
    }
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }
  console.error("Timed out waiting for cloud connection");
  return false;
}

interface UseStreamLifecycleParams {
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  startStream: (
    initialParams?: Record<string, unknown>,
    stream?: MediaStream
  ) => void;
  stopStream: () => void;
  isStreaming: boolean;
  sessionId: string | null;
  loadPipeline: (
    pipelineIds?: string[],
    loadParams?: Record<string, unknown>
  ) => Promise<boolean>;
  localStream: MediaStream | null;
  videoResolution: { width: number; height: number } | null;
  mode: VideoSourceMode;
}

export function useStreamLifecycle(params: UseStreamLifecycleParams) {
  const {
    startStream,
    stopStream,
    isStreaming,
    sessionId,
    loadPipeline,
    localStream,
    videoResolution,
    mode,
  } = params;

  const api = useApi();
  const { pipelines } = usePipelinesContext();
  const store = useAppStore;

  const handleStartStream = useCallback(
    async (overridePipelineId?: PipelineId): Promise<boolean> => {
      if (isStreaming) {
        stopStream();
        return true;
      }

      const { settings, isRecording } = store.getState();
      const pipelineIdToUse = overridePipelineId || settings.pipelineId;

      try {
        const pipelineIds: string[] = [];
        if (settings.preprocessorIds && settings.preprocessorIds.length > 0) {
          pipelineIds.push(...settings.preprocessorIds);
        }
        pipelineIds.push(pipelineIdToUse);
        if (settings.postprocessorIds && settings.postprocessorIds.length > 0) {
          pipelineIds.push(...settings.postprocessorIds);
        }

        // Check model downloads
        const missingPipelines: string[] = [];
        for (const pid of pipelineIds) {
          const info = pipelines?.[pid];
          if (info?.requiresModels) {
            try {
              const status = await api.checkModelStatus(pid);
              if (!status.downloaded) {
                missingPipelines.push(pid);
              }
            } catch (error) {
              console.error(`Error checking model status for ${pid}:`, error);
            }
          }
        }

        if (missingPipelines.length > 0) {
          store.getState().setPipelinesNeedingModels(missingPipelines);
          store.getState().setShowDownloadDialog(true);
          return false;
        }

        // Wait for cloud connection if needed
        try {
          const cloudRes = await fetch("/api/v1/cloud/status");
          if (cloudRes.ok) {
            const cloudData = await cloudRes.json();
            if (cloudData.connecting && !cloudData.connected) {
              console.log(
                "[StreamPage] Cloud connecting, waiting before pipeline load..."
              );
              store.getState().setIsCloudConnecting(true);
              try {
                const cloudReady = await waitForCloudConnection();
                if (!cloudReady) {
                  console.error("Cloud connection failed, cannot start stream");
                  return false;
                }
              } finally {
                store.getState().setIsCloudConnecting(false);
              }
            }
          }
        } catch (e) {
          console.error("Error checking cloud status before stream:", e);
        }

        console.log(`Loading ${pipelineIdToUse} pipeline...`);

        const getPipelineDefaultMode = (pid: string): InputMode =>
          pipelines?.[pid]?.defaultMode ?? "text";

        const currentMode =
          settings.inputMode ||
          getPipelineDefaultMode(pipelineIdToUse) ||
          "text";

        let resolution = settings.resolution || videoResolution;

        if (resolution) {
          const { resolution: adjustedResolution, wasAdjusted } =
            adjustResolutionForPipeline(pipelineIdToUse, resolution);

          if (wasAdjusted) {
            store.getState().updateSettings({ resolution: adjustedResolution });
            resolution = adjustedResolution;
          }
        }

        const currentPipeline = pipelines?.[pipelineIdToUse];
        const vaceEnabled = currentPipeline?.supportsVACE
          ? (settings.vaceEnabled ?? currentMode !== "video")
          : false;

        let loadParams: Record<string, unknown> | null = null;

        if (resolution) {
          loadParams = {
            height: resolution.height,
            width: resolution.width,
          };

          if (currentPipeline?.supportsQuantization) {
            loadParams.quantization = settings.quantization ?? null;
          }

          if (currentPipeline?.supportsLoRA && settings.loras) {
            const loraParams = buildLoRAParams(
              settings.loras,
              settings.loraMergeStrategy
            );
            loadParams = { ...loadParams, ...loraParams };
          }

          if (currentPipeline?.supportsVACE) {
            loadParams.vace_enabled = vaceEnabled;
            const vaceParams = getVaceParams(
              settings.refImages,
              settings.vaceContextScale
            );
            loadParams = { ...loadParams, ...vaceParams };
          }

          if (
            settings.schemaFieldOverrides &&
            Object.keys(settings.schemaFieldOverrides).length > 0
          ) {
            loadParams = { ...loadParams, ...settings.schemaFieldOverrides };
          }

          console.log(
            `Loading ${pipelineIds.length} pipeline(s) (${pipelineIds.join(", ")}) with resolution ${resolution.width}x${resolution.height}`,
            loadParams
          );
        }

        const loadSuccess = await loadPipeline(
          pipelineIds,
          loadParams || undefined
        );
        if (!loadSuccess) {
          console.error("Failed to load pipeline, cannot start stream");
          return false;
        }

        const needsVideoInput = currentMode === "video";
        const isSpoutMode = mode === "spout" && settings.spoutReceiver?.enabled;

        const streamToSend =
          needsVideoInput && !isSpoutMode
            ? localStream || undefined
            : undefined;

        if (needsVideoInput && !isSpoutMode && !localStream) {
          console.error("Video input required but no local stream available");
          return false;
        }

        const { promptItems, interpolationMethod } = store.getState();

        const initialParameters: Record<string, unknown> = {
          input_mode: currentMode,
        };

        if (currentPipeline?.supportsPrompts !== false) {
          initialParameters.prompts = promptItems;
          initialParameters.prompt_interpolation_method = interpolationMethod;
          initialParameters.denoising_step_list = settings.denoisingSteps || [
            700, 500,
          ];
        }

        if (currentPipeline?.supportsCacheManagement) {
          initialParameters.manage_cache = settings.manageCache ?? true;
        }

        if (currentPipeline?.supportsKvCacheBias) {
          initialParameters.kv_cache_attention_bias =
            settings.kvCacheAttentionBias ?? 1.0;
        }

        initialParameters.pipeline_ids = pipelineIds;

        if (currentPipeline?.supportsVACE) {
          const vaceParams = getVaceParams(
            settings.refImages,
            settings.vaceContextScale
          );
          if ("vace_ref_images" in vaceParams) {
            initialParameters.vace_ref_images = vaceParams.vace_ref_images;
            initialParameters.vace_context_scale =
              vaceParams.vace_context_scale;
          }
          if (currentMode === "video") {
            initialParameters.vace_use_input_video =
              settings.vaceUseInputVideo ?? false;
          }
          initialParameters.vace_enabled = vaceEnabled;
        } else if (
          currentPipeline?.supportsImages &&
          settings.refImages?.length
        ) {
          initialParameters.images = settings.refImages;
        }

        if (settings.firstFrameImage) {
          initialParameters.first_frame_image = settings.firstFrameImage;
        }
        if (settings.lastFrameImage) {
          initialParameters.last_frame_image = settings.lastFrameImage;
        }

        if (currentMode === "video") {
          initialParameters.noise_scale = settings.noiseScale ?? 0.7;
          initialParameters.noise_controller = settings.noiseController ?? true;
        }

        if (settings.spoutSender?.enabled) {
          initialParameters.spout_sender = settings.spoutSender;
        }
        if (settings.spoutReceiver?.enabled) {
          initialParameters.spout_receiver = settings.spoutReceiver;
        }

        initialParameters.recording = isRecording;

        if (
          settings.schemaFieldOverrides &&
          Object.keys(settings.schemaFieldOverrides).length > 0
        ) {
          Object.assign(initialParameters, settings.schemaFieldOverrides);
        }

        store.getState().updateSettings({ paused: false });

        startStream(initialParameters, streamToSend);

        return true;
      } catch (error) {
        console.error("Error during stream start:", error);
        return false;
      }
    },
    [
      isStreaming,
      stopStream,
      pipelines,
      api,
      loadPipeline,
      localStream,
      videoResolution,
      mode,
      startStream,
    ]
  );

  const handleSaveGeneration = useCallback(async () => {
    try {
      if (!sessionId) {
        toast.error("No active session", {
          description: "Please start a stream before downloading the recording",
          duration: 5000,
        });
        return;
      }
      await api.downloadRecording(sessionId);
    } catch (error) {
      console.error("Error downloading recording:", error);
      toast.error("Error downloading recording", {
        description:
          error instanceof Error
            ? error.message
            : "An error occurred while downloading the recording",
        duration: 5000,
      });
    }
  }, [sessionId, api]);

  return {
    handleStartStream,
    handleSaveGeneration,
  };
}
