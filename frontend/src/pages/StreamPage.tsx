import { useState, useEffect, useRef } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { PromptInputWithTimeline } from "../components/PromptInputWithTimeline";
import { DownloadDialog } from "../components/DownloadDialog";
import type { TimelinePrompt } from "../components/PromptTimeline";
import { StatusBar } from "../components/StatusBar";
import { useWebRTC } from "../hooks/useWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { useControllerInput } from "../hooks/useControllerInput";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { usePipelines } from "../hooks/usePipelines";
import { getDefaultPromptForMode } from "../data/pipelines";
import { adjustResolutionForPipeline } from "../lib/utils";
import type {
  ExtensionMode,
  InputMode,
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  DownloadProgress,
  VaeType,
} from "../types";
import type { PromptItem, PromptTransition } from "../lib/api";
import {
  checkModelStatus,
  downloadPipelineModels,
  downloadRecording,
} from "../lib/api";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";
import { toast } from "sonner";

// Delay before resetting video reinitialization flag (ms)
// This allows useVideoSource to detect the flag change and trigger reinitialization
const VIDEO_REINITIALIZE_DELAY_MS = 100;

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

export function StreamPage() {
  // Fetch available pipelines dynamically
  const { pipelines } = usePipelines();

  // Helper to get default mode for a pipeline
  const getPipelineDefaultMode = (pipelineId: string): InputMode => {
    return pipelines?.[pipelineId]?.defaultMode ?? "text";
  };

  // Use the stream state hook for settings management
  const {
    settings,
    updateSettings,
    getDefaults,
    supportsNoiseControls,
    spoutAvailable,
  } = useStreamState();

  // Prompt state - use unified default prompts based on mode
  const initialMode =
    settings.inputMode || getPipelineDefaultMode(settings.pipelineId);
  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: getDefaultPromptForMode(initialMode), weight: 100 },
  ]);
  const [interpolationMethod, setInterpolationMethod] = useState<
    "linear" | "slerp"
  >("linear");
  const [temporalInterpolationMethod, setTemporalInterpolationMethod] =
    useState<"linear" | "slerp">("slerp");
  const [transitionSteps, setTransitionSteps] = useState(4);

  // Track when we need to reinitialize video source
  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

  // Store custom video resolution from user uploads - persists across mode/pipeline changes
  const [customVideoResolution, setCustomVideoResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const [isLive, setIsLive] = useState(false);
  const [isTimelineCollapsed, setIsTimelineCollapsed] = useState(false);
  const [selectedTimelinePrompt, setSelectedTimelinePrompt] =
    useState<TimelinePrompt | null>(null);

  // Timeline state for left panel
  const [timelinePrompts, setTimelinePrompts] = useState<TimelinePrompt[]>([]);
  const [timelineCurrentTime, setTimelineCurrentTime] = useState(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);

  // Video display state
  const [videoScaleMode, setVideoScaleMode] = useState<"fit" | "native">("fit");

  // External control of timeline selection
  const [externalSelectedPromptId, setExternalSelectedPromptId] = useState<
    string | null
  >(null);

  // Download dialog state
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [pipelinesNeedingModels, setPipelinesNeedingModels] = useState<
    string[]
  >([]);
  const [currentDownloadPipeline, setCurrentDownloadPipeline] = useState<
    string | null
  >(null);

  // Ref to access timeline functions
  const timelineRef = useRef<{
    getCurrentTimelinePrompt: () => string;
    submitLivePrompt: (prompts: PromptItem[]) => void;
    updatePrompt: (prompt: TimelinePrompt) => void;
    clearTimeline: () => void;
    resetPlayhead: () => void;
    resetTimelineCompletely: () => void;
    getPrompts: () => TimelinePrompt[];
    getCurrentTime: () => number;
    getIsPlaying: () => boolean;
  }>(null);

  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
    pipelineInfo,
  } = usePipeline();

  // WebRTC for streaming
  const {
    remoteStream,
    isStreaming,
    isConnecting,
    peerConnectionRef,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate,
    sessionId,
  } = useWebRTC();

  // Computed loading state - true when downloading models, loading pipeline, or connecting WebRTC
  const isLoading = isDownloading || isPipelineLoading || isConnecting;

  // Get WebRTC stats for FPS
  const webrtcStats = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
  });

  // Video container ref for controller input pointer lock
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // Check if current pipeline supports controller input
  const currentPipelineSupportsController =
    pipelines?.[settings.pipelineId]?.supportsControllerInput ?? false;

  // Controller input hook - captures WASD/mouse and streams to backend
  const { isPointerLocked, requestPointerLock } = useControllerInput(
    sendParameterUpdate,
    isStreaming && currentPipelineSupportsController,
    videoContainerRef
  );

  // Video source for preview (camera or video)
  // Enable based on input mode, not pipeline category
  const {
    localStream,
    isInitializing,
    error: videoSourceError,
    mode,
    videoResolution,
    switchMode,
    handleVideoFileUpload,
  } = useVideoSource({
    onStreamUpdate: updateVideoTrack,
    onStopStream: stopStream,
    shouldReinitialize: shouldReinitializeVideo,
    enabled: settings.inputMode === "video",
    // Sync output resolution when user uploads a custom video
    // Store the custom resolution so it persists across mode/pipeline changes
    onCustomVideoResolution: resolution => {
      setCustomVideoResolution(resolution);
      updateSettings({
        resolution: { height: resolution.height, width: resolution.width },
      });
    },
  });

  const handlePromptsSubmit = (prompts: PromptItem[]) => {
    setPromptItems(prompts);
  };

  const handleTransitionSubmit = (transition: PromptTransition) => {
    setPromptItems(transition.target_prompts);

    // Add to timeline if available
    if (timelineRef.current) {
      timelineRef.current.submitLivePrompt(transition.target_prompts);
    }

    // Send transition to backend
    sendParameterUpdate({
      transition,
    });
  };

  // Handler for input mode changes (text vs video)
  const handleInputModeChange = (newMode: InputMode) => {
    // Stop stream if currently streaming
    if (isStreaming) {
      stopStream();
    }

    // Get mode-specific defaults from backend schema
    const modeDefaults = getDefaults(settings.pipelineId, newMode);

    // Use custom video resolution if switching to video mode and one exists
    // This preserves the user's uploaded video resolution across mode switches
    const resolution =
      newMode === "video" && customVideoResolution
        ? customVideoResolution
        : { height: modeDefaults.height, width: modeDefaults.width };

    // Update settings with new mode and ALL mode-specific defaults including resolution
    updateSettings({
      inputMode: newMode,
      resolution,
      denoisingSteps: modeDefaults.denoisingSteps,
      noiseScale: modeDefaults.noiseScale,
      noiseController: modeDefaults.noiseController,
    });

    // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
    setPromptItems([{ text: getDefaultPromptForMode(newMode), weight: 100 }]);

    // Update temporal interpolation steps to mode-specific default
    const pipeline = pipelines?.[settings.pipelineId];
    const pipelineDefaultSteps =
      pipeline?.defaultTemporalInterpolationSteps ?? 4;
    setTransitionSteps(
      modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps
    );

    // Handle video source based on mode
    if (newMode === "video") {
      // Trigger video source reinitialization
      setShouldReinitializeVideo(true);
      setTimeout(
        () => setShouldReinitializeVideo(false),
        VIDEO_REINITIALIZE_DELAY_MS
      );
    }
    // Note: useVideoSource hook will automatically stop when enabled becomes false
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
    // Stop the stream if it's currently running
    if (isStreaming) {
      stopStream();
    }

    const newPipeline = pipelines?.[pipelineId];
    const modeToUse = newPipeline?.defaultMode || "text";
    const currentMode = settings.inputMode || "text";

    // Trigger video reinitialization if switching to video mode
    if (modeToUse === "video" && currentMode !== "video") {
      setShouldReinitializeVideo(true);
      setTimeout(
        () => setShouldReinitializeVideo(false),
        VIDEO_REINITIALIZE_DELAY_MS
      );
    }

    // Reset timeline completely but preserve collapse state
    if (timelineRef.current) {
      timelineRef.current.resetTimelineCompletely();
    }

    // Reset selected timeline prompt to exit Edit mode and return to Append mode
    setSelectedTimelinePrompt(null);
    setExternalSelectedPromptId(null);

    // Get all defaults for the new pipeline + mode from backend schema
    const defaults = getDefaults(pipelineId, modeToUse);

    // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
    setPromptItems([{ text: getDefaultPromptForMode(modeToUse), weight: 100 }]);

    // Use custom video resolution if mode is video and one exists
    // This preserves the user's uploaded video resolution across pipeline switches
    const resolution =
      modeToUse === "video" && customVideoResolution
        ? customVideoResolution
        : { height: defaults.height, width: defaults.width };

    // Update the pipeline in settings with the appropriate mode and defaults
    updateSettings({
      pipelineId,
      inputMode: modeToUse,
      denoisingSteps: defaults.denoisingSteps,
      resolution,
      noiseScale: defaults.noiseScale,
      noiseController: defaults.noiseController,
      loras: [], // Clear LoRA controls when switching pipelines
    });
  };

  const downloadPipelineSequentially = async (
    pipelineId: string,
    remainingPipelines: string[]
  ) => {
    setCurrentDownloadPipeline(pipelineId);
    setDownloadProgress(null);

    try {
      await downloadPipelineModels(pipelineId);

      // Enhanced polling with progress updates
      const checkDownloadProgress = async () => {
        try {
          const status = await checkModelStatus(pipelineId);

          // Update progress state
          if (status.progress) {
            setDownloadProgress(status.progress);
          }

          if (status.downloaded) {
            // Download complete for this pipeline
            // Remove it from the list
            const newRemaining = remainingPipelines;
            setPipelinesNeedingModels(newRemaining);

            // Check if this was a preprocessor or the main pipeline
            const pipelineInfo = pipelines?.[pipelineId];
            const isPreprocessor =
              pipelineInfo?.usage?.includes("preprocessor") ?? false;

            // Only update the main pipeline ID if this was NOT a preprocessor
            // and it matches the current pipeline ID
            if (!isPreprocessor && pipelineId === settings.pipelineId) {
              // This is the main pipeline, update settings
              if (timelineRef.current) {
                timelineRef.current.resetTimelineCompletely();
              }

              setSelectedTimelinePrompt(null);
              setExternalSelectedPromptId(null);

              // Preserve the current input mode that the user selected before download
              const newPipeline = pipelines?.[pipelineId];
              const currentMode =
                settings.inputMode || newPipeline?.defaultMode || "text";
              const defaults = getDefaults(
                pipelineId as PipelineId,
                currentMode
              );

              // Use custom video resolution if mode is video and one exists
              const resolution =
                currentMode === "video" && customVideoResolution
                  ? customVideoResolution
                  : { height: defaults.height, width: defaults.width };

              // Only update pipeline-related settings, preserving current input mode and prompts
              updateSettings({
                pipelineId: pipelineId as PipelineId,
                inputMode: currentMode,
                denoisingSteps: defaults.denoisingSteps,
                resolution,
                noiseScale: defaults.noiseScale,
                noiseController: defaults.noiseController,
              });
            }

            // If there are more pipelines to download, continue with the next one
            if (newRemaining.length > 0) {
              // Continue with next pipeline
              setTimeout(() => {
                downloadPipelineSequentially(
                  newRemaining[0],
                  newRemaining.slice(1)
                );
              }, 1000);
            } else {
              // All downloads complete
              setIsDownloading(false);
              setDownloadProgress(null);
              setShowDownloadDialog(false);
              setCurrentDownloadPipeline(null);

              // Automatically start the stream after all downloads complete
              setTimeout(async () => {
                const started = await handleStartStream();
                // If stream started successfully, also start the timeline
                if (started && timelinePlayPauseRef.current) {
                  setTimeout(() => {
                    timelinePlayPauseRef.current?.();
                  }, 2000); // Give stream time to fully initialize
                }
              }, 100);
            }
          } else {
            setTimeout(checkDownloadProgress, 2000);
          }
        } catch (error) {
          console.error("Error checking download status:", error);
          setIsDownloading(false);
          setDownloadProgress(null);
          setShowDownloadDialog(false);
          setCurrentDownloadPipeline(null);
        }
      };

      // Start checking
      setTimeout(checkDownloadProgress, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      setIsDownloading(false);
      setDownloadProgress(null);
      setShowDownloadDialog(false);
      setCurrentDownloadPipeline(null);
    }
  };

  const handleDownloadModels = async () => {
    if (pipelinesNeedingModels.length === 0) return;

    setIsDownloading(true);
    setShowDownloadDialog(true); // Keep dialog open to show progress

    // Start downloading the first pipeline in the list
    const firstPipeline = pipelinesNeedingModels[0];
    const remaining = pipelinesNeedingModels.slice(1);
    await downloadPipelineSequentially(firstPipeline, remaining);
  };

  const handleDialogClose = () => {
    setShowDownloadDialog(false);
    setPipelinesNeedingModels([]);
    setCurrentDownloadPipeline(null);

    // When user cancels, no stream or timeline has started yet, so nothing to clean up
    // Just close the dialog and return early without any state changes
  };

  const handleResolutionChange = (resolution: {
    height: number;
    width: number;
  }) => {
    updateSettings({ resolution });
  };

  const handleSeedChange = (seed: number) => {
    updateSettings({ seed });
  };

  const handleDenoisingStepsChange = (denoisingSteps: number[]) => {
    updateSettings({ denoisingSteps });
    // Send denoising steps update to backend
    sendParameterUpdate({
      denoising_step_list: denoisingSteps,
    });
  };

  const handleNoiseScaleChange = (noiseScale: number) => {
    updateSettings({ noiseScale });
    // Send noise scale update to backend
    sendParameterUpdate({
      noise_scale: noiseScale,
    });
  };

  const handleNoiseControllerChange = (enabled: boolean) => {
    updateSettings({ noiseController: enabled });
    // Send noise controller update to backend
    sendParameterUpdate({
      noise_controller: enabled,
    });
  };

  const handleManageCacheChange = (enabled: boolean) => {
    updateSettings({ manageCache: enabled });
    // Send manage cache update to backend
    sendParameterUpdate({
      manage_cache: enabled,
    });
  };

  const handleQuantizationChange = (quantization: "fp8_e4m3fn" | null) => {
    updateSettings({ quantization });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleVaeTypeChange = (vaeType: VaeType) => {
    updateSettings({ vaeType });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleKvCacheAttentionBiasChange = (bias: number) => {
    updateSettings({ kvCacheAttentionBias: bias });
    // Send KV cache attention bias update to backend
    sendParameterUpdate({
      kv_cache_attention_bias: bias,
    });
  };

  const handleLorasChange = (loras: LoRAConfig[]) => {
    updateSettings({ loras });

    // If streaming, send scale updates to backend for runtime adjustment
    if (isStreaming) {
      sendLoRAScaleUpdates(
        loras,
        pipelineInfo?.loaded_lora_adapters,
        ({ lora_scales }) => {
          // Forward only the lora_scales field over the data channel.
          sendParameterUpdate({
            // TypeScript doesn't know about lora_scales on this payload yet.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ...({ lora_scales } as any),
          });
        }
      );
    }
    // Note: Adding/removing LoRAs requires pipeline reload
  };

  const handleVaceEnabledChange = (enabled: boolean) => {
    updateSettings({ vaceEnabled: enabled });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleVaceUseInputVideoChange = (enabled: boolean) => {
    updateSettings({ vaceUseInputVideo: enabled });
    // Send parameter update to backend if streaming
    if (isStreaming) {
      sendParameterUpdate({
        vace_use_input_video: enabled,
      });
    }
  };

  const handleRefImagesChange = (images: string[]) => {
    updateSettings({ refImages: images });
  };

  const handleSendHints = (imagePaths: string[]) => {
    const currentPipeline = pipelines?.[settings.pipelineId];

    if (currentPipeline?.supportsVACE) {
      // VACE pipeline - use vace_ref_images
      sendParameterUpdate({
        vace_ref_images: imagePaths,
      });
    } else if (currentPipeline?.supportsImages) {
      // Non-VACE pipeline with images support - use images
      sendParameterUpdate({
        images: imagePaths,
      });
    }
  };

  const handlePreprocessorIdsChange = (ids: string[]) => {
    updateSettings({ preprocessorIds: ids });
  };

  const handlePostprocessorIdsChange = (ids: string[]) => {
    updateSettings({ postprocessorIds: ids });
  };

  const handleVaceContextScaleChange = (scale: number) => {
    updateSettings({ vaceContextScale: scale });
    // Send VACE context scale update to backend if streaming
    if (isStreaming) {
      sendParameterUpdate({
        vace_context_scale: scale,
      });
    }
  };

  // Derive the appropriate extension mode based on which frame images are set
  const deriveExtensionMode = (
    first: string | undefined,
    last: string | undefined
  ): ExtensionMode | undefined => {
    if (first && last) return "firstlastframe";
    if (first) return "firstframe";
    if (last) return "lastframe";
    return undefined;
  };

  const handleFirstFrameImageChange = (imagePath: string | undefined) => {
    updateSettings({
      firstFrameImage: imagePath,
      extensionMode: deriveExtensionMode(imagePath, settings.lastFrameImage),
    });
  };

  const handleLastFrameImageChange = (imagePath: string | undefined) => {
    updateSettings({
      lastFrameImage: imagePath,
      extensionMode: deriveExtensionMode(settings.firstFrameImage, imagePath),
    });
  };

  const handleExtensionModeChange = (mode: ExtensionMode) => {
    updateSettings({ extensionMode: mode });
  };

  const handleSendExtensionFrames = () => {
    const mode = settings.extensionMode || "firstframe";
    const params: Record<string, string> = {};

    if (mode === "firstframe" && settings.firstFrameImage) {
      params.first_frame_image = settings.firstFrameImage;
    } else if (mode === "lastframe" && settings.lastFrameImage) {
      params.last_frame_image = settings.lastFrameImage;
    } else if (mode === "firstlastframe") {
      if (settings.firstFrameImage) {
        params.first_frame_image = settings.firstFrameImage;
      }
      if (settings.lastFrameImage) {
        params.last_frame_image = settings.lastFrameImage;
      }
    }

    if (Object.keys(params).length > 0) {
      sendParameterUpdate(params);
    }
  };

  const handleResetCache = () => {
    // Send reset cache command to backend
    sendParameterUpdate({
      reset_cache: true,
    });
  };

  const handleSpoutSenderChange = (
    spoutSender: { enabled: boolean; name: string } | undefined
  ) => {
    updateSettings({ spoutSender });
    // Send Spout output settings to backend
    if (isStreaming) {
      sendParameterUpdate({
        spout_sender: spoutSender,
      });
    }
  };

  // Handle Spout input name change from InputAndControlsPanel
  const handleSpoutReceiverChange = (name: string) => {
    updateSettings({
      spoutReceiver: {
        enabled: mode === "spout",
        name: name,
      },
    });
  };

  // Sync spoutReceiver.enabled with mode changes
  const handleModeChange = (newMode: typeof mode) => {
    // When switching to spout mode, enable spout input
    if (newMode === "spout") {
      updateSettings({
        spoutReceiver: {
          enabled: true,
          name: settings.spoutReceiver?.name ?? "",
        },
      });
    } else {
      // When switching away from spout mode, disable spout input
      updateSettings({
        spoutReceiver: {
          enabled: false,
          name: settings.spoutReceiver?.name ?? "",
        },
      });
    }
    switchMode(newMode);
  };

  const handleLivePromptSubmit = (prompts: PromptItem[]) => {
    // Use the timeline ref to submit the prompt
    if (timelineRef.current) {
      timelineRef.current.submitLivePrompt(prompts);
    }

    // Also send the updated parameters to the backend immediately
    // Preserve the full blend while live
    sendParameterUpdate({
      prompts,
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: settings.denoisingSteps || [700, 500],
    });
  };

  const handleTimelinePromptEdit = (prompt: TimelinePrompt | null) => {
    setSelectedTimelinePrompt(prompt);
    // Sync external selection state
    setExternalSelectedPromptId(prompt?.id || null);
  };

  const handleTimelinePromptUpdate = (prompt: TimelinePrompt) => {
    setSelectedTimelinePrompt(prompt);

    // Update the prompt in the timeline
    if (timelineRef.current) {
      timelineRef.current.updatePrompt(prompt);
    }
  };

  // Event-driven timeline state updates for left panel
  const handleTimelinePromptsChange = (prompts: TimelinePrompt[]) => {
    setTimelinePrompts(prompts);
  };

  const handleTimelineCurrentTimeChange = (currentTime: number) => {
    setTimelineCurrentTime(currentTime);
  };

  const handleTimelinePlayingChange = (isPlaying: boolean) => {
    setIsTimelinePlaying(isPlaying);
  };

  // Handle ESC key to exit Edit mode and return to Append mode
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selectedTimelinePrompt) {
        setSelectedTimelinePrompt(null);
        setExternalSelectedPromptId(null);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedTimelinePrompt]);

  // Update temporal interpolation defaults and clear prompts when pipeline changes
  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline) {
      const defaultMethod =
        pipeline.defaultTemporalInterpolationMethod || "slerp";
      const pipelineDefaultSteps =
        pipeline.defaultTemporalInterpolationSteps ?? 4;
      // Get mode-specific default if available
      const modeDefaults = getDefaults(settings.pipelineId, settings.inputMode);
      const defaultSteps =
        modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps;

      setTemporalInterpolationMethod(defaultMethod);
      setTransitionSteps(defaultSteps);

      // Clear prompts if pipeline doesn't support them
      if (pipeline.supportsPrompts === false) {
        setPromptItems([{ text: "", weight: 1.0 }]);
      }
    }
  }, [settings.pipelineId, pipelines, settings.inputMode, getDefaults]);

  const handlePlayPauseToggle = () => {
    const newPausedState = !settings.paused;
    updateSettings({ paused: newPausedState });
    sendParameterUpdate({
      paused: newPausedState,
    });

    // Deselect any selected prompt when video starts playing
    if (!newPausedState && selectedTimelinePrompt) {
      setSelectedTimelinePrompt(null);
      setExternalSelectedPromptId(null); // Also clear external selection
    }
  };

  // Ref to access the timeline's play/pause handler
  const timelinePlayPauseRef = useRef<(() => Promise<void>) | null>(null);

  // Ref to store callback that should execute when video starts playing
  const onVideoPlayingCallbackRef = useRef<(() => void) | null>(null);

  // Note: We intentionally do NOT auto-sync videoResolution to settings.resolution.
  // Mode defaults from the backend schema take precedence. Users can manually
  // adjust resolution if needed. This prevents the video source resolution from
  // overriding the carefully tuned per-mode defaults.

  const handleStartStream = async (
    overridePipelineId?: PipelineId
  ): Promise<boolean> => {
    if (isStreaming) {
      stopStream();
      return true;
    }

    // Use override pipeline ID if provided, otherwise use current settings
    const pipelineIdToUse = overridePipelineId || settings.pipelineId;

    try {
      // Build pipeline chain: preprocessors + main pipeline + postprocessors
      const pipelineIds: string[] = [];
      if (settings.preprocessorIds && settings.preprocessorIds.length > 0) {
        pipelineIds.push(...settings.preprocessorIds);
      }
      pipelineIds.push(pipelineIdToUse);
      if (settings.postprocessorIds && settings.postprocessorIds.length > 0) {
        pipelineIds.push(...settings.postprocessorIds);
      }

      // Check if models are needed but not downloaded for all pipelines in the chain
      // Collect all missing pipelines/preprocessors
      const missingPipelines: string[] = [];
      for (const pipelineId of pipelineIds) {
        const pipelineInfo = pipelines?.[pipelineId];
        if (pipelineInfo?.requiresModels) {
          try {
            const status = await checkModelStatus(pipelineId);
            if (!status.downloaded) {
              missingPipelines.push(pipelineId);
            }
          } catch (error) {
            console.error(
              `Error checking model status for ${pipelineId}:`,
              error
            );
            // Continue anyway if check fails
          }
        }
      }

      // If any pipelines are missing models, show download dialog
      if (missingPipelines.length > 0) {
        setPipelinesNeedingModels(missingPipelines);
        setShowDownloadDialog(true);
        return false; // Stream did not start
      }

      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${pipelineIdToUse} pipeline...`);

      // Determine current input mode
      const currentMode =
        settings.inputMode || getPipelineDefaultMode(pipelineIdToUse) || "text";

      // Use settings.resolution if available, otherwise fall back to videoResolution
      let resolution = settings.resolution || videoResolution;

      // Adjust resolution to be divisible by required scale factor for the pipeline
      if (resolution) {
        const { resolution: adjustedResolution, wasAdjusted } =
          adjustResolutionForPipeline(pipelineIdToUse, resolution);

        if (wasAdjusted) {
          // Update settings with adjusted resolution
          updateSettings({ resolution: adjustedResolution });
          resolution = adjustedResolution;
        }
      }

      // Build load parameters dynamically based on pipeline capabilities and settings
      // The backend will use only the parameters it needs based on the pipeline schema
      const currentPipeline = pipelines?.[pipelineIdToUse];
      // Compute VACE enabled state - needed for both loadParams and initialParameters
      const vaceEnabled = currentPipeline?.supportsVACE
        ? (settings.vaceEnabled ?? currentMode !== "video")
        : false;

      let loadParams: Record<string, unknown> | null = null;

      if (resolution) {
        // Start with common parameters
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };

        // Add seed if pipeline supports quantization (implies it needs seed)
        if (currentPipeline?.supportsQuantization) {
          loadParams.seed = settings.seed ?? 42;
          loadParams.quantization = settings.quantization ?? null;
          loadParams.vae_type = settings.vaeType ?? "wan";
        }

        // Add LoRA parameters if pipeline supports LoRA
        if (currentPipeline?.supportsLoRA && settings.loras) {
          const loraParams = buildLoRAParams(
            settings.loras,
            settings.loraMergeStrategy
          );
          loadParams = { ...loadParams, ...loraParams };
        }

        // Add VACE parameters if pipeline supports VACE
        if (currentPipeline?.supportsVACE) {
          loadParams.vace_enabled = vaceEnabled;

          // Add VACE reference images if provided
          const vaceParams = getVaceParams(
            settings.refImages,
            settings.vaceContextScale
          );
          loadParams = { ...loadParams, ...vaceParams };
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

      // Check video requirements based on input mode
      const needsVideoInput = currentMode === "video";
      const isSpoutMode = mode === "spout" && settings.spoutReceiver?.enabled;

      // Only send video stream for pipelines that need video input (not in Spout mode)
      const streamToSend =
        needsVideoInput && !isSpoutMode ? localStream || undefined : undefined;

      if (needsVideoInput && !isSpoutMode && !localStream) {
        console.error("Video input required but no local stream available");
        return false;
      }

      // Build initial parameters based on pipeline type
      const initialParameters: {
        input_mode?: "text" | "video";
        prompts?: PromptItem[];
        prompt_interpolation_method?: "linear" | "slerp";
        denoising_step_list?: number[];
        noise_scale?: number;
        noise_controller?: boolean;
        manage_cache?: boolean;
        kv_cache_attention_bias?: number;
        spout_sender?: { enabled: boolean; name: string };
        spout_receiver?: { enabled: boolean; name: string };
        vace_ref_images?: string[];
        vace_use_input_video?: boolean;
        vace_context_scale?: number;
        vace_enabled?: boolean;
        pipeline_ids?: string[];
        first_frame_image?: string;
        last_frame_image?: string;
        images?: string[];
      } = {
        // Signal the intended input mode to the backend so it doesn't
        // briefly fall back to text mode before video frames arrive
        input_mode: currentMode,
      };

      // Common parameters for pipelines that support prompts
      if (currentPipeline?.supportsPrompts !== false) {
        initialParameters.prompts = promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = settings.denoisingSteps || [
          700, 500,
        ];
      }

      // Cache management for pipelines that support it
      if (currentPipeline?.supportsCacheManagement) {
        initialParameters.manage_cache = settings.manageCache ?? true;
      }

      // KV cache bias for pipelines that support it
      if (currentPipeline?.supportsKvCacheBias) {
        initialParameters.kv_cache_attention_bias =
          settings.kvCacheAttentionBias ?? 1.0;
      }

      // Pipeline chain: preprocessors + main pipeline (already built above)
      initialParameters.pipeline_ids = pipelineIds;

      // VACE-specific parameters
      if (currentPipeline?.supportsVACE) {
        const vaceParams = getVaceParams(
          settings.refImages,
          settings.vaceContextScale
        );
        if ("vace_ref_images" in vaceParams) {
          initialParameters.vace_ref_images = vaceParams.vace_ref_images;
          initialParameters.vace_context_scale = vaceParams.vace_context_scale;
        }
        // Add vace_use_input_video parameter
        if (currentMode === "video") {
          initialParameters.vace_use_input_video =
            settings.vaceUseInputVideo ?? false;
        }
        initialParameters.vace_enabled = vaceEnabled;
      } else if (
        currentPipeline?.supportsImages &&
        settings.refImages?.length
      ) {
        // Non-VACE pipelines that support images
        initialParameters.images = settings.refImages;
      }

      // Add FFLF (first-frame-last-frame) parameters if set
      if (settings.firstFrameImage) {
        initialParameters.first_frame_image = settings.firstFrameImage;
      }
      if (settings.lastFrameImage) {
        initialParameters.last_frame_image = settings.lastFrameImage;
      }

      // Video mode parameters - applies to all pipelines in video mode
      if (currentMode === "video") {
        initialParameters.noise_scale = settings.noiseScale ?? 0.7;
        initialParameters.noise_controller = settings.noiseController ?? true;
      }

      // Spout settings - send if enabled
      if (settings.spoutSender?.enabled) {
        initialParameters.spout_sender = settings.spoutSender;
      }
      if (settings.spoutReceiver?.enabled) {
        initialParameters.spout_receiver = settings.spoutReceiver;
      }

      // Reset paused state when starting a fresh stream
      updateSettings({ paused: false });

      // Pipeline is loaded, now start WebRTC stream
      startStream(initialParameters, streamToSend);

      return true; // Stream started successfully
    } catch (error) {
      console.error("Error during stream start:", error);
      return false;
    }
  };

  const handleSaveGeneration = async () => {
    try {
      if (!sessionId) {
        toast.error("No active session", {
          description: "Please start a stream before downloading the recording",
          duration: 5000,
        });
        return;
      }
      await downloadRecording(sessionId);
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
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <Header />

      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 px-4 pb-4 min-h-0 overflow-hidden">
        {/* Left Panel - Input & Controls */}
        <div className="w-1/5">
          <InputAndControlsPanel
            className="h-full"
            pipelines={pipelines}
            localStream={localStream}
            isInitializing={isInitializing}
            error={videoSourceError}
            mode={mode}
            onModeChange={handleModeChange}
            isStreaming={isStreaming}
            isConnecting={isConnecting}
            isPipelineLoading={isPipelineLoading}
            canStartStream={
              settings.inputMode === "text"
                ? !isInitializing
                : mode === "spout"
                  ? !isInitializing // Spout mode doesn't need local stream
                  : !!localStream && !isInitializing
            }
            onStartStream={handleStartStream}
            onStopStream={stopStream}
            onVideoFileUpload={handleVideoFileUpload}
            pipelineId={settings.pipelineId}
            prompts={promptItems}
            onPromptsChange={setPromptItems}
            onPromptsSubmit={handlePromptsSubmit}
            onTransitionSubmit={handleTransitionSubmit}
            interpolationMethod={interpolationMethod}
            onInterpolationMethodChange={setInterpolationMethod}
            temporalInterpolationMethod={temporalInterpolationMethod}
            onTemporalInterpolationMethodChange={setTemporalInterpolationMethod}
            isLive={isLive}
            onLivePromptSubmit={handleLivePromptSubmit}
            selectedTimelinePrompt={selectedTimelinePrompt}
            onTimelinePromptUpdate={handleTimelinePromptUpdate}
            isVideoPaused={settings.paused}
            isTimelinePlaying={isTimelinePlaying}
            currentTime={timelineCurrentTime}
            timelinePrompts={timelinePrompts}
            transitionSteps={transitionSteps}
            onTransitionStepsChange={setTransitionSteps}
            spoutReceiverName={settings.spoutReceiver?.name ?? ""}
            onSpoutReceiverChange={handleSpoutReceiverChange}
            inputMode={
              settings.inputMode || getPipelineDefaultMode(settings.pipelineId)
            }
            onInputModeChange={handleInputModeChange}
            spoutAvailable={spoutAvailable}
            vaceEnabled={
              settings.vaceEnabled ??
              (pipelines?.[settings.pipelineId]?.supportsVACE &&
                settings.inputMode !== "video")
            }
            refImages={settings.refImages || []}
            onRefImagesChange={handleRefImagesChange}
            onSendHints={handleSendHints}
            isDownloading={isDownloading}
            supportsImages={pipelines?.[settings.pipelineId]?.supportsImages}
            firstFrameImage={settings.firstFrameImage}
            onFirstFrameImageChange={handleFirstFrameImageChange}
            lastFrameImage={settings.lastFrameImage}
            onLastFrameImageChange={handleLastFrameImageChange}
            extensionMode={settings.extensionMode || "firstframe"}
            onExtensionModeChange={handleExtensionModeChange}
            onSendExtensionFrames={handleSendExtensionFrames}
          />
        </div>

        {/* Center Panel - Video Output + Timeline */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Video area - takes remaining space but can shrink */}
          <div className="flex-1 min-h-0">
            <VideoOutput
              className="h-full"
              remoteStream={remoteStream}
              isPipelineLoading={isPipelineLoading}
              isConnecting={isConnecting}
              pipelineError={pipelineError}
              isPlaying={!settings.paused}
              isDownloading={isDownloading}
              onPlayPauseToggle={() => {
                // Use timeline's play/pause handler instead of direct video toggle
                if (timelinePlayPauseRef.current) {
                  timelinePlayPauseRef.current();
                }
              }}
              onStartStream={() => {
                // Use timeline's play/pause handler to start stream
                if (timelinePlayPauseRef.current) {
                  timelinePlayPauseRef.current();
                }
              }}
              onVideoPlaying={() => {
                // Execute callback when video starts playing
                if (onVideoPlayingCallbackRef.current) {
                  onVideoPlayingCallbackRef.current();
                  onVideoPlayingCallbackRef.current = null; // Clear after execution
                }
              }}
              // Controller input props
              supportsControllerInput={currentPipelineSupportsController}
              isPointerLocked={isPointerLocked}
              onRequestPointerLock={requestPointerLock}
              videoContainerRef={videoContainerRef}
              // Video scale mode
              videoScaleMode={videoScaleMode}
            />
          </div>
          {/* Timeline area - compact, always visible */}
          <div className="flex-shrink-0 mt-2">
            <PromptInputWithTimeline
              currentPrompt={promptItems[0]?.text || ""}
              currentPromptItems={promptItems}
              transitionSteps={transitionSteps}
              temporalInterpolationMethod={temporalInterpolationMethod}
              onPromptSubmit={text => {
                // Update the left panel's prompt state to reflect current timeline prompt
                const prompts = [{ text, weight: 100 }];
                setPromptItems(prompts);

                // Send to backend - use transition if streaming and transition steps > 0
                if (isStreaming && transitionSteps > 0) {
                  sendParameterUpdate({
                    transition: {
                      target_prompts: prompts,
                      num_steps: transitionSteps,
                      temporal_interpolation_method:
                        temporalInterpolationMethod,
                    },
                  });
                } else {
                  // Send direct prompts without transition
                  sendParameterUpdate({
                    prompts,
                    prompt_interpolation_method: interpolationMethod,
                    denoising_step_list: settings.denoisingSteps || [700, 500],
                  });
                }
              }}
              onPromptItemsSubmit={(
                prompts,
                blockTransitionSteps,
                blockTemporalInterpolationMethod
              ) => {
                // Update the left panel's prompt state to reflect current timeline prompt blend
                setPromptItems(prompts);

                // Use transition params from block if provided, otherwise use global settings
                const effectiveTransitionSteps =
                  blockTransitionSteps ?? transitionSteps;
                const effectiveTemporalInterpolationMethod =
                  blockTemporalInterpolationMethod ??
                  temporalInterpolationMethod;

                // Update the left panel's transition settings to reflect current block's values
                if (blockTransitionSteps !== undefined) {
                  setTransitionSteps(blockTransitionSteps);
                }
                if (blockTemporalInterpolationMethod !== undefined) {
                  setTemporalInterpolationMethod(
                    blockTemporalInterpolationMethod
                  );
                }

                // Send to backend - use transition if streaming and transition steps > 0
                if (isStreaming && effectiveTransitionSteps > 0) {
                  sendParameterUpdate({
                    transition: {
                      target_prompts: prompts,
                      num_steps: effectiveTransitionSteps,
                      temporal_interpolation_method:
                        effectiveTemporalInterpolationMethod,
                    },
                  });
                } else {
                  // Send direct prompts without transition
                  sendParameterUpdate({
                    prompts,
                    prompt_interpolation_method: interpolationMethod,
                    denoising_step_list: settings.denoisingSteps || [700, 500],
                  });
                }
              }}
              disabled={isPipelineLoading || isConnecting || showDownloadDialog}
              isStreaming={isStreaming}
              isVideoPaused={settings.paused}
              timelineRef={timelineRef}
              onLiveStateChange={setIsLive}
              onLivePromptSubmit={handleLivePromptSubmit}
              onDisconnect={stopStream}
              onStartStream={handleStartStream}
              onVideoPlayPauseToggle={handlePlayPauseToggle}
              onPromptEdit={handleTimelinePromptEdit}
              isCollapsed={isTimelineCollapsed}
              onCollapseToggle={setIsTimelineCollapsed}
              externalSelectedPromptId={externalSelectedPromptId}
              settings={settings}
              onSettingsImport={updateSettings}
              onPlayPauseRef={timelinePlayPauseRef}
              onVideoPlayingCallbackRef={onVideoPlayingCallbackRef}
              onResetCache={handleResetCache}
              onTimelinePromptsChange={handleTimelinePromptsChange}
              onTimelineCurrentTimeChange={handleTimelineCurrentTimeChange}
              onTimelinePlayingChange={handleTimelinePlayingChange}
              isLoading={isLoading}
              videoScaleMode={videoScaleMode}
              onVideoScaleModeToggle={() =>
                setVideoScaleMode(prev => (prev === "fit" ? "native" : "fit"))
              }
              isDownloading={isDownloading}
              onSaveGeneration={handleSaveGeneration}
            />
          </div>
        </div>

        {/* Right Panel - Settings */}
        <div className="w-1/5">
          <SettingsPanel
            className="h-full"
            pipelines={pipelines}
            pipelineId={settings.pipelineId}
            onPipelineIdChange={handlePipelineIdChange}
            isStreaming={isStreaming}
            isLoading={isLoading}
            resolution={
              settings.resolution || {
                height: getDefaults(settings.pipelineId, settings.inputMode)
                  .height,
                width: getDefaults(settings.pipelineId, settings.inputMode)
                  .width,
              }
            }
            onResolutionChange={handleResolutionChange}
            seed={settings.seed ?? 42}
            onSeedChange={handleSeedChange}
            denoisingSteps={
              settings.denoisingSteps ||
              getDefaults(settings.pipelineId, settings.inputMode)
                .denoisingSteps || [750, 250]
            }
            onDenoisingStepsChange={handleDenoisingStepsChange}
            defaultDenoisingSteps={
              getDefaults(settings.pipelineId, settings.inputMode)
                .denoisingSteps || [750, 250]
            }
            noiseScale={settings.noiseScale ?? 0.7}
            onNoiseScaleChange={handleNoiseScaleChange}
            noiseController={settings.noiseController ?? true}
            onNoiseControllerChange={handleNoiseControllerChange}
            manageCache={settings.manageCache ?? true}
            onManageCacheChange={handleManageCacheChange}
            quantization={
              settings.quantization !== undefined
                ? settings.quantization
                : "fp8_e4m3fn"
            }
            onQuantizationChange={handleQuantizationChange}
            kvCacheAttentionBias={settings.kvCacheAttentionBias ?? 0.3}
            onKvCacheAttentionBiasChange={handleKvCacheAttentionBiasChange}
            onResetCache={handleResetCache}
            loras={settings.loras || []}
            onLorasChange={handleLorasChange}
            loraMergeStrategy={settings.loraMergeStrategy ?? "permanent_merge"}
            inputMode={settings.inputMode}
            supportsNoiseControls={supportsNoiseControls(settings.pipelineId)}
            spoutSender={settings.spoutSender}
            onSpoutSenderChange={handleSpoutSenderChange}
            spoutAvailable={spoutAvailable}
            vaceEnabled={
              settings.vaceEnabled ??
              (pipelines?.[settings.pipelineId]?.supportsVACE &&
                settings.inputMode !== "video")
            }
            onVaceEnabledChange={handleVaceEnabledChange}
            vaceUseInputVideo={settings.vaceUseInputVideo ?? false}
            onVaceUseInputVideoChange={handleVaceUseInputVideoChange}
            vaceContextScale={settings.vaceContextScale ?? 1.0}
            onVaceContextScaleChange={handleVaceContextScaleChange}
            vaeType={settings.vaeType ?? "wan"}
            onVaeTypeChange={handleVaeTypeChange}
            vaeTypes={pipelines?.[settings.pipelineId]?.vaeTypes}
            preprocessorIds={settings.preprocessorIds ?? []}
            onPreprocessorIdsChange={handlePreprocessorIdsChange}
            postprocessorIds={settings.postprocessorIds ?? []}
            onPostprocessorIdsChange={handlePostprocessorIdsChange}
          />
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar fps={webrtcStats.fps} bitrate={webrtcStats.bitrate} />

      {/* Download Dialog */}
      {pipelinesNeedingModels.length > 0 && (
        <DownloadDialog
          open={showDownloadDialog}
          pipelines={pipelines}
          pipelineIds={pipelinesNeedingModels}
          currentDownloadPipeline={currentDownloadPipeline}
          onClose={handleDialogClose}
          onDownload={handleDownloadModels}
          isDownloading={isDownloading}
          progress={downloadProgress}
        />
      )}
    </div>
  );
}
