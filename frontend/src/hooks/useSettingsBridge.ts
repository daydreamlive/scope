/**
 * Bridge handlers that sync Zustand store settings with WebRTC parameter updates.
 *
 * Each handler follows the pattern: update store → send parameter to WebRTC peer.
 */

import { useCallback } from "react";
import { useAppStore } from "../stores";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";
import type { ExtensionMode, LoRAConfig } from "../types";
import type { PipelineStatusResponse } from "../lib/api";
import type { VideoSourceMode } from "./useVideoSource";

interface UseSettingsBridgeParams {
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  isStreaming: boolean;
  pipelineInfo: PipelineStatusResponse | null;
  mode: VideoSourceMode;
}

export function useSettingsBridge(params: UseSettingsBridgeParams) {
  const { sendParameterUpdate, isStreaming, pipelineInfo, mode } = params;

  const { pipelines } = usePipelinesContext();
  const store = useAppStore;

  const handleResolutionChange = useCallback(
    (dimension: "height" | "width", value: number) => {
      const { settings } = store.getState();
      const resolution = settings.resolution || { height: 512, width: 512 };
      store.getState().updateSettings({
        resolution: { ...resolution, [dimension]: value },
      });
    },
    []
  );

  const handleDenoisingStepsChange = useCallback(
    (steps: number[]) => {
      store.getState().updateSettings({ denoisingSteps: steps });
      sendParameterUpdate({ denoising_step_list: steps });
    },
    [sendParameterUpdate]
  );

  const handleNoiseScaleChange = useCallback(
    (value: number) => {
      store.getState().updateSettings({ noiseScale: value });
      sendParameterUpdate({ noise_scale: value });
    },
    [sendParameterUpdate]
  );

  const handleNoiseControllerChange = useCallback(
    (enabled: boolean) => {
      store.getState().updateSettings({ noiseController: enabled });
      sendParameterUpdate({ noise_controller: enabled });
    },
    [sendParameterUpdate]
  );

  const handleManageCacheChange = useCallback(
    (enabled: boolean) => {
      store.getState().updateSettings({ manageCache: enabled });
      sendParameterUpdate({ manage_cache: enabled });
    },
    [sendParameterUpdate]
  );

  const handleResetCache = useCallback(() => {
    sendParameterUpdate({ reset_cache: true });
  }, [sendParameterUpdate]);

  const handleQuantizationChange = useCallback((q: "fp8_e4m3fn" | null) => {
    store.getState().updateSettings({ quantization: q });
  }, []);

  const handleKvCacheAttentionBiasChange = useCallback(
    (value: number) => {
      store.getState().updateSettings({ kvCacheAttentionBias: value });
      sendParameterUpdate({ kv_cache_attention_bias: value });
    },
    [sendParameterUpdate]
  );

  const handleLorasChange = useCallback(
    (loras: LoRAConfig[]) => {
      store.getState().updateSettings({ loras });
      if (isStreaming) {
        sendLoRAScaleUpdates(
          loras,
          pipelineInfo?.loaded_lora_adapters,
          ({ lora_scales }) => {
            sendParameterUpdate({
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              ...({ lora_scales } as any),
            });
          }
        );
      }
    },
    [isStreaming, pipelineInfo, sendParameterUpdate]
  );

  const handleVaceEnabledChange = useCallback((enabled: boolean) => {
    store.getState().updateSettings({ vaceEnabled: enabled });
  }, []);

  const handleVaceUseInputVideoChange = useCallback(
    (enabled: boolean) => {
      store.getState().updateSettings({ vaceUseInputVideo: enabled });
      if (isStreaming) sendParameterUpdate({ vace_use_input_video: enabled });
    },
    [isStreaming, sendParameterUpdate]
  );

  const handleVaceContextScaleChange = useCallback(
    (value: number) => {
      store.getState().updateSettings({ vaceContextScale: value });
      if (isStreaming) sendParameterUpdate({ vace_context_scale: value });
    },
    [isStreaming, sendParameterUpdate]
  );

  const handleRefImagesChange = useCallback((images: string[]) => {
    store.getState().updateSettings({ refImages: images });
  }, []);

  const handleSendHints = useCallback(
    (imagePaths: string[]) => {
      const { settings } = store.getState();
      const currentPipeline = pipelines?.[settings.pipelineId];
      if (currentPipeline?.supportsVACE) {
        sendParameterUpdate({ vace_ref_images: imagePaths });
      } else if (currentPipeline?.supportsImages) {
        sendParameterUpdate({ images: imagePaths });
      }
    },
    [pipelines, sendParameterUpdate]
  );

  const handlePreprocessorIdsChange = useCallback((ids: string[]) => {
    store.getState().updateSettings({ preprocessorIds: ids });
  }, []);

  const handlePostprocessorIdsChange = useCallback((ids: string[]) => {
    store.getState().updateSettings({ postprocessorIds: ids });
  }, []);

  const handleFirstFrameImageChange = useCallback(
    (imagePath: string | undefined) => {
      const { settings } = store.getState();
      const lastFrame = settings.lastFrameImage;
      let extMode: ExtensionMode | undefined;
      if (imagePath && lastFrame) extMode = "firstlastframe";
      else if (imagePath) extMode = "firstframe";
      else if (lastFrame) extMode = "lastframe";
      store.getState().updateSettings({
        firstFrameImage: imagePath,
        extensionMode: extMode,
      });
    },
    []
  );

  const handleLastFrameImageChange = useCallback(
    (imagePath: string | undefined) => {
      const { settings } = store.getState();
      const firstFrame = settings.firstFrameImage;
      let extMode: ExtensionMode | undefined;
      if (firstFrame && imagePath) extMode = "firstlastframe";
      else if (firstFrame) extMode = "firstframe";
      else if (imagePath) extMode = "lastframe";
      store.getState().updateSettings({
        lastFrameImage: imagePath,
        extensionMode: extMode,
      });
    },
    []
  );

  const handleExtensionModeChange = useCallback((extMode: ExtensionMode) => {
    store.getState().updateSettings({ extensionMode: extMode });
  }, []);

  const handleSendExtensionFrames = useCallback(() => {
    const { settings } = store.getState();
    const extMode = settings.extensionMode || "firstframe";
    const sendParams: Record<string, string> = {};
    if (extMode === "firstframe" && settings.firstFrameImage) {
      sendParams.first_frame_image = settings.firstFrameImage;
    } else if (extMode === "lastframe" && settings.lastFrameImage) {
      sendParams.last_frame_image = settings.lastFrameImage;
    } else if (extMode === "firstlastframe") {
      if (settings.firstFrameImage)
        sendParams.first_frame_image = settings.firstFrameImage;
      if (settings.lastFrameImage)
        sendParams.last_frame_image = settings.lastFrameImage;
    }
    if (Object.keys(sendParams).length > 0) {
      sendParameterUpdate(sendParams);
    }
  }, [sendParameterUpdate]);

  const handleSpoutSenderChange = useCallback(
    (sender: { enabled: boolean; name: string }) => {
      store.getState().updateSettings({ spoutSender: sender });
      if (isStreaming) sendParameterUpdate({ spout_sender: sender });
    },
    [isStreaming, sendParameterUpdate]
  );

  const handleSpoutReceiverChange = useCallback(
    (name: string) => {
      store.getState().updateSettings({
        spoutReceiver: { enabled: mode === "spout", name },
      });
    },
    [mode]
  );

  const handleSchemaFieldOverrideChange = useCallback(
    (key: string, value: unknown, isRuntimeParam?: boolean) => {
      const { settings } = store.getState();
      store.getState().updateSettings({
        schemaFieldOverrides: {
          ...(settings.schemaFieldOverrides ?? {}),
          [key]: value,
        },
      });
      if (isRuntimeParam && isStreaming) sendParameterUpdate({ [key]: value });
    },
    [isStreaming, sendParameterUpdate]
  );

  const handlePlayPauseToggle = useCallback(() => {
    const { settings } = store.getState();
    const paused = settings.paused ?? false;
    store.getState().updateSettings({ paused: !paused });
    sendParameterUpdate({ paused: !paused });
    if (paused) {
      // Unpausing — clear timeline selection
      store.getState().setSelectedTimelinePrompt(null);
      store.getState().setExternalSelectedPromptId(null);
    }
  }, [sendParameterUpdate]);

  return {
    handleResolutionChange,
    handleDenoisingStepsChange,
    handleNoiseScaleChange,
    handleNoiseControllerChange,
    handleManageCacheChange,
    handleResetCache,
    handleQuantizationChange,
    handleKvCacheAttentionBiasChange,
    handleLorasChange,
    handleVaceEnabledChange,
    handleVaceUseInputVideoChange,
    handleVaceContextScaleChange,
    handleRefImagesChange,
    handleSendHints,
    handlePreprocessorIdsChange,
    handlePostprocessorIdsChange,
    handleFirstFrameImageChange,
    handleLastFrameImageChange,
    handleExtensionModeChange,
    handleSendExtensionFrames,
    handleSpoutSenderChange,
    handleSpoutReceiverChange,
    handleSchemaFieldOverrideChange,
    handlePlayPauseToggle,
  };
}
