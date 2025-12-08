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
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import {
  PIPELINES,
  getPipelineDefaultMode,
  getDefaultPromptForMode,
} from "../data/pipelines";
import type {
  InputMode,
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
} from "../types";
import type { PromptItem, PromptTransition } from "../lib/api";
import { checkModelStatus, downloadPipelineModels } from "../lib/api";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";

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

export function StreamPage() {
  // Use the stream state hook for settings management
  const { settings, updateSettings, getDefaults, supportsNoiseControls } =
    useStreamState();

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

  // External control of timeline selection
  const [externalSelectedPromptId, setExternalSelectedPromptId] = useState<
    string | null
  >(null);

  // Download dialog state
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [pipelineNeedsModels, setPipelineNeedsModels] = useState<string | null>(
    null
  );

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
  } = useWebRTC();

  // Get WebRTC stats for FPS
  const webrtcStats = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
  });

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

    const newPipeline = PIPELINES[pipelineId];
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

  const handleDownloadModels = async () => {
    if (!pipelineNeedsModels) return;

    setIsDownloading(true);
    setShowDownloadDialog(false);

    try {
      await downloadPipelineModels(pipelineNeedsModels);

      // Start polling to check when download is complete
      const checkDownloadComplete = async () => {
        try {
          const status = await checkModelStatus(pipelineNeedsModels);
          if (status.downloaded) {
            setIsDownloading(false);
            setPipelineNeedsModels(null);

            // Now update the pipeline since download is complete
            const pipelineId = pipelineNeedsModels as PipelineId;

            if (timelineRef.current) {
              timelineRef.current.resetTimelineCompletely();
            }

            setSelectedTimelinePrompt(null);
            setExternalSelectedPromptId(null);

            // Get defaults for the pipeline's default mode
            const newPipeline = PIPELINES[pipelineId];
            const defaultMode = newPipeline?.defaultMode || "text";
            const defaults = getDefaults(pipelineId, defaultMode);

            // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
            setPromptItems([
              { text: getDefaultPromptForMode(defaultMode), weight: 100 },
            ]);

            // Use custom video resolution if mode is video and one exists
            const resolution =
              defaultMode === "video" && customVideoResolution
                ? customVideoResolution
                : { height: defaults.height, width: defaults.width };

            updateSettings({
              pipelineId,
              inputMode: defaultMode,
              denoisingSteps: defaults.denoisingSteps,
              resolution,
              noiseScale: defaults.noiseScale,
              noiseController: defaults.noiseController,
            });

            // Automatically start the stream after download completes
            // Use setTimeout to ensure state updates are processed first
            setTimeout(async () => {
              const started = await handleStartStream();
              // If stream started successfully, also start the timeline
              if (started && timelinePlayPauseRef.current) {
                setTimeout(() => {
                  timelinePlayPauseRef.current?.();
                }, 2000); // Give stream time to fully initialize
              }
            }, 100);
          } else {
            // Check again in 2 seconds
            setTimeout(checkDownloadComplete, 2000);
          }
        } catch (error) {
          console.error("Error checking download status:", error);
          setIsDownloading(false);
        }
      };

      // Start checking for completion
      setTimeout(checkDownloadComplete, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      setIsDownloading(false);
    }
  };

  const handleDialogClose = () => {
    setShowDownloadDialog(false);
    setPipelineNeedsModels(null);

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

  const handleResetCache = () => {
    // Send reset cache command to backend
    sendParameterUpdate({
      reset_cache: true,
    });
  };

  const handleSpoutOutputChange = (
    spoutOutput: { enabled: boolean; senderName: string } | undefined
  ) => {
    updateSettings({ spoutOutput });
    // Send Spout output settings to backend
    if (isStreaming) {
      sendParameterUpdate({
        spout_output: spoutOutput,
      });
    }
  };

  // Handle Spout input name change from InputAndControlsPanel
  const handleSpoutInputNameChange = (name: string) => {
    updateSettings({
      spoutInput: {
        enabled: mode === "spout",
        senderName: name,
      },
    });
  };

  // Sync spoutInput.enabled with mode changes
  const handleModeChange = (newMode: typeof mode) => {
    // When switching to spout mode, enable spout input
    if (newMode === "spout") {
      updateSettings({
        spoutInput: {
          enabled: true,
          senderName: settings.spoutInput?.senderName ?? "",
        },
      });
    } else {
      // When switching away from spout mode, disable spout input
      updateSettings({
        spoutInput: {
          enabled: false,
          senderName: settings.spoutInput?.senderName ?? "",
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

  // Update temporal interpolation defaults when pipeline changes
  useEffect(() => {
    const pipeline = PIPELINES[settings.pipelineId];
    if (pipeline) {
      const defaultMethod =
        pipeline.defaultTemporalInterpolationMethod || "slerp";
      const defaultSteps = pipeline.defaultTemporalInterpolationSteps ?? 4;

      setTemporalInterpolationMethod(defaultMethod);
      setTransitionSteps(defaultSteps);
    }
  }, [settings.pipelineId]);

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
      // Check if models are needed but not downloaded
      const pipelineInfo = PIPELINES[pipelineIdToUse];
      if (pipelineInfo?.requiresModels) {
        try {
          const status = await checkModelStatus(pipelineIdToUse);
          if (!status.downloaded) {
            // Show download dialog
            setPipelineNeedsModels(pipelineIdToUse);
            setShowDownloadDialog(true);
            return false; // Stream did not start
          }
        } catch (error) {
          console.error("Error checking model status:", error);
          // Continue anyway if check fails
        }
      }

      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${pipelineIdToUse} pipeline...`);

      // Determine current input mode
      const currentMode =
        settings.inputMode || getPipelineDefaultMode(pipelineIdToUse) || "text";

      // Prepare load parameters based on pipeline type
      let loadParams = null;

      // Use settings.resolution if available, otherwise fall back to videoResolution
      const resolution = settings.resolution || videoResolution;

      if (pipelineIdToUse === "streamdiffusionv2" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
          quantization: settings.quantization ?? null,
          ...buildLoRAParams(settings.loras, settings.loraMergeStrategy),
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}, quantization: ${loadParams.quantization}, lora_merge_mode: ${loadParams.lora_merge_mode}`
        );
      } else if (pipelineIdToUse === "passthrough" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}`
        );
      } else if (pipelineIdToUse === "longlive" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
          quantization: settings.quantization ?? null,
          ...buildLoRAParams(settings.loras, settings.loraMergeStrategy),
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}, quantization: ${loadParams.quantization}, lora_merge_mode: ${loadParams.lora_merge_mode}`
        );
      } else if (settings.pipelineId === "krea-realtime-video" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
          quantization:
            settings.quantization !== undefined
              ? settings.quantization
              : "fp8_e4m3fn",
          ...buildLoRAParams(settings.loras, settings.loraMergeStrategy),
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}, quantization: ${loadParams.quantization}, lora_merge_mode: ${loadParams.lora_merge_mode}`
        );
      } else if (pipelineIdToUse === "reward-forcing" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
          quantization: settings.quantization ?? null,
          ...buildLoRAParams(settings.loras, settings.loraMergeStrategy),
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}, quantization: ${loadParams.quantization}, lora_merge_mode: ${loadParams.lora_merge_mode}`
        );
      }

      const loadSuccess = await loadPipeline(
        pipelineIdToUse,
        loadParams || undefined
      );
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        return false;
      }

      // Check video requirements based on input mode
      const needsVideoInput = currentMode === "video";
      const isSpoutMode = mode === "spout" && settings.spoutInput?.enabled;

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
        spout_output?: { enabled: boolean; senderName: string };
        spout_input?: { enabled: boolean; senderName: string };
      } = {
        // Signal the intended input mode to the backend so it doesn't
        // briefly fall back to text mode before video frames arrive
        input_mode: currentMode,
      };

      // Common parameters for pipelines that support prompts
      if (pipelineIdToUse !== "passthrough") {
        initialParameters.prompts = promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = settings.denoisingSteps || [
          700, 500,
        ];
      }

      // Cache management for krea_realtime_video, longlive, and reward-forcing
      if (
        pipelineIdToUse === "krea-realtime-video" ||
        pipelineIdToUse === "longlive" ||
        pipelineIdToUse === "reward-forcing"
      ) {
        initialParameters.manage_cache = settings.manageCache ?? true;
      }

      // Krea-realtime-video-specific parameters
      if (pipelineIdToUse === "krea-realtime-video") {
        initialParameters.kv_cache_attention_bias =
          settings.kvCacheAttentionBias ?? 1.0;
      }

      // Video mode parameters - applies to all pipelines in video mode
      if (currentMode === "video") {
        initialParameters.noise_scale = settings.noiseScale ?? 0.7;
        initialParameters.noise_controller = settings.noiseController ?? true;
      }

      // Spout settings - send if enabled
      if (settings.spoutOutput?.enabled) {
        initialParameters.spout_output = settings.spoutOutput;
      }
      if (settings.spoutInput?.enabled) {
        initialParameters.spout_input = settings.spoutInput;
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
            spoutInputName={settings.spoutInput?.senderName ?? ""}
            onSpoutInputNameChange={handleSpoutInputNameChange}
            inputMode={
              settings.inputMode || getPipelineDefaultMode(settings.pipelineId)
            }
            onInputModeChange={handleInputModeChange}
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
              disabled={
                settings.pipelineId === "passthrough" ||
                isPipelineLoading ||
                isConnecting ||
                showDownloadDialog
              }
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
              isDownloading={isDownloading}
            />
          </div>
        </div>

        {/* Right Panel - Settings */}
        <div className="w-1/5">
          <SettingsPanel
            className="h-full"
            pipelineId={settings.pipelineId}
            onPipelineIdChange={handlePipelineIdChange}
            isStreaming={isStreaming}
            isDownloading={isDownloading}
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
            spoutOutput={settings.spoutOutput}
            onSpoutOutputChange={handleSpoutOutputChange}
          />
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar fps={webrtcStats.fps} bitrate={webrtcStats.bitrate} />

      {/* Download Dialog */}
      {pipelineNeedsModels && (
        <DownloadDialog
          open={showDownloadDialog}
          pipelineId={pipelineNeedsModels as PipelineId}
          onClose={handleDialogClose}
          onDownload={handleDownloadModels}
        />
      )}
    </div>
  );
}
