/**
 * Pipeline, mode, prompt, and timeline orchestration handlers.
 *
 * Manages pipeline switching, input mode changes, prompt submission,
 * and timeline prompt editing.
 */

import { useCallback } from "react";
import { useAppStore } from "../stores";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamState } from "./useStreamState";
import { getDefaultPromptForMode } from "../data/pipelines";
import type { InputMode, PipelineId } from "../types";
import type { VideoSourceMode } from "./useVideoSource";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { TimelinePrompt } from "../components/PromptTimeline";

// Delay before resetting video reinitialization flag (ms)
const VIDEO_REINITIALIZE_DELAY_MS = 100;

interface UsePipelineHandlersParams {
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  isStreaming: boolean;
  stopStream: () => void;
  switchMode: (mode: VideoSourceMode) => void | Promise<void>;
  timelineRef: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
    submitLivePrompt: (prompts: PromptItem[]) => void;
    updatePrompt: (prompt: TimelinePrompt) => void;
    clearTimeline: () => void;
    resetPlayhead: () => void;
    resetTimelineCompletely: () => void;
    getPrompts: () => TimelinePrompt[];
    getCurrentTime: () => number;
    getIsPlaying: () => boolean;
  } | null>;
}

export function usePipelineHandlers(params: UsePipelineHandlersParams) {
  const {
    sendParameterUpdate,
    isStreaming,
    stopStream,
    switchMode,
    timelineRef,
  } = params;

  const { pipelines, refreshPipelines } = usePipelinesContext();
  const {
    getDefaults,
    supportsNoiseControls,
    refreshPipelineSchemas,
    refreshHardwareInfo,
  } = useStreamState();

  const store = useAppStore;

  // ---- Combined refresh ----
  const handlePipelinesRefresh = useCallback(async () => {
    await Promise.all([
      refreshPipelineSchemas(),
      refreshPipelines(),
      refreshHardwareInfo(),
    ]);
  }, [refreshPipelineSchemas, refreshPipelines, refreshHardwareInfo]);

  // ---- Prompt handlers ----
  const handleTransitionSubmit = useCallback(
    (transition: PromptTransition) => {
      store.getState().setPromptItems(transition.target_prompts);

      if (timelineRef.current) {
        timelineRef.current.submitLivePrompt(transition.target_prompts);
      }

      sendParameterUpdate({ transition });
    },
    [sendParameterUpdate, timelineRef]
  );

  const handleLivePromptSubmit = useCallback(
    (prompts: PromptItem[]) => {
      if (timelineRef.current) {
        timelineRef.current.submitLivePrompt(prompts);
      }

      const { interpolationMethod, settings } = store.getState();
      sendParameterUpdate({
        prompts,
        prompt_interpolation_method: interpolationMethod,
        denoising_step_list: settings.denoisingSteps || [700, 500],
      });
    },
    [sendParameterUpdate, timelineRef]
  );

  // ---- Input mode / pipeline change ----
  const handleInputModeChange = useCallback(
    (newMode: InputMode) => {
      if (isStreaming) stopStream();

      const { settings, customVideoResolution } = store.getState();
      const modeDefaults = getDefaults(settings.pipelineId, newMode);

      const resolution =
        newMode === "video" && customVideoResolution
          ? customVideoResolution
          : { height: modeDefaults.height, width: modeDefaults.width };

      store.getState().updateSettings({
        inputMode: newMode,
        resolution,
        denoisingSteps: modeDefaults.denoisingSteps,
        noiseScale: modeDefaults.noiseScale,
        noiseController: modeDefaults.noiseController,
      });

      store
        .getState()
        .setPromptItems([
          { text: getDefaultPromptForMode(newMode), weight: 100 },
        ]);

      const pipeline = pipelines?.[settings.pipelineId];
      const pipelineDefaultSteps =
        pipeline?.defaultTemporalInterpolationSteps ?? 4;
      store
        .getState()
        .setTransitionSteps(
          modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps
        );

      if (newMode === "video") {
        store.getState().setShouldReinitializeVideo(true);
        setTimeout(
          () => store.getState().setShouldReinitializeVideo(false),
          VIDEO_REINITIALIZE_DELAY_MS
        );
      }
    },
    [isStreaming, stopStream, getDefaults, pipelines]
  );

  const handlePipelineIdChange = useCallback(
    (pipelineId: PipelineId) => {
      if (isStreaming) stopStream();

      const newPipeline = pipelines?.[pipelineId];
      const modeToUse = newPipeline?.defaultMode || "text";
      const { settings, customVideoResolution } = store.getState();
      const currentMode = settings.inputMode || "text";

      if (modeToUse === "video" && currentMode !== "video") {
        store.getState().setShouldReinitializeVideo(true);
        setTimeout(
          () => store.getState().setShouldReinitializeVideo(false),
          VIDEO_REINITIALIZE_DELAY_MS
        );
      }

      if (timelineRef.current) {
        timelineRef.current.resetTimelineCompletely();
      }

      store.getState().setSelectedTimelinePrompt(null);
      store.getState().setExternalSelectedPromptId(null);

      const defaults = getDefaults(pipelineId, modeToUse);

      store
        .getState()
        .setPromptItems([
          { text: getDefaultPromptForMode(modeToUse), weight: 100 },
        ]);

      const resolution =
        modeToUse === "video" && customVideoResolution
          ? customVideoResolution
          : { height: defaults.height, width: defaults.width };

      store.getState().updateSettings({
        pipelineId,
        inputMode: modeToUse,
        denoisingSteps: defaults.denoisingSteps,
        resolution,
        noiseScale: defaults.noiseScale,
        noiseController: defaults.noiseController,
        loras: [],
      });
    },
    [isStreaming, stopStream, pipelines, getDefaults, timelineRef]
  );

  const handleModeChange = useCallback(
    (newMode: VideoSourceMode) => {
      const { settings } = store.getState();
      if (newMode === "spout") {
        store.getState().updateSettings({
          spoutReceiver: {
            enabled: true,
            name: settings.spoutReceiver?.name ?? "",
          },
        });
      } else {
        store.getState().updateSettings({
          spoutReceiver: {
            enabled: false,
            name: settings.spoutReceiver?.name ?? "",
          },
        });
      }
      switchMode(newMode);
    },
    [switchMode]
  );

  // ---- Timeline prompt handlers ----
  const handleTimelinePromptUpdate = useCallback(
    (prompt: TimelinePrompt) => {
      store.getState().setSelectedTimelinePrompt(prompt);
      if (timelineRef.current) {
        timelineRef.current.updatePrompt(prompt);
      }
    },
    [timelineRef]
  );

  const handleTimelinePromptEdit = useCallback(
    (prompt: TimelinePrompt | null) => {
      store.getState().setSelectedTimelinePrompt(prompt);
      store.getState().setExternalSelectedPromptId(prompt?.id || null);
    },
    []
  );

  return {
    // Refresh
    handlePipelinesRefresh,

    // Prompts
    handleTransitionSubmit,
    handleLivePromptSubmit,

    // Mode/pipeline
    handleInputModeChange,
    handlePipelineIdChange,
    handleModeChange,

    // Timeline
    handleTimelinePromptUpdate,
    handleTimelinePromptEdit,

    // Derived (from useStreamState)
    supportsNoiseControls,
    getDefaults,
  };
}
