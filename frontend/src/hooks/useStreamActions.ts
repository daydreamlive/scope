/**
 * Orchestrator hook that composes all stream action sub-hooks.
 *
 * Bridges Zustand store state with WebRTC parameter updates and
 * other imperative actions (pipeline loading, video source, timeline).
 */

import { useRef } from "react";
import { useSettingsBridge } from "./useSettingsBridge";
import { usePipelineHandlers } from "./usePipelineHandlers";
import { useDownloadHandlers } from "./useDownloadHandlers";
import { useStreamLifecycle } from "./useStreamLifecycle";
import type { PipelineId } from "../types";
import type { PipelineStatusResponse } from "../lib/api";
import type { VideoSourceMode } from "./useVideoSource";
import type { PromptItem } from "../lib/api";
import type { TimelinePrompt } from "../components/PromptTimeline";

interface UseStreamActionsParams {
  // WebRTC functions (imperative, can't be in store)
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  startStream: (
    initialParams?: Record<string, unknown>,
    stream?: MediaStream
  ) => void;
  stopStream: () => void;
  isStreaming: boolean;
  sessionId: string | null;

  // Pipeline hook
  loadPipeline: (
    pipelineIds?: string[],
    loadParams?: Record<string, unknown>
  ) => Promise<boolean>;
  pipelineInfo: PipelineStatusResponse | null;

  // Video source
  localStream: MediaStream | null;
  videoResolution: { width: number; height: number } | null;
  mode: VideoSourceMode;
  switchMode: (mode: VideoSourceMode) => void | Promise<void>;

  // Refs (imperative timeline access)
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
  timelinePlayPauseRef: React.RefObject<(() => Promise<void>) | null>;
}

export function useStreamActions(params: UseStreamActionsParams) {
  const {
    sendParameterUpdate,
    startStream,
    stopStream,
    isStreaming,
    sessionId,
    loadPipeline,
    pipelineInfo,
    localStream,
    videoResolution,
    mode,
    switchMode,
    timelineRef,
    timelinePlayPauseRef,
  } = params;

  // Settings bridge handlers
  const settingsBridge = useSettingsBridge({
    sendParameterUpdate,
    isStreaming,
    pipelineInfo,
    mode,
  });

  // Pipeline/mode/prompt/timeline handlers
  const pipelineHandlers = usePipelineHandlers({
    sendParameterUpdate,
    isStreaming,
    stopStream,
    switchMode,
    timelineRef,
  });

  // Stream lifecycle (start/stop/save)
  const streamLifecycle = useStreamLifecycle({
    sendParameterUpdate,
    startStream,
    stopStream,
    isStreaming,
    sessionId,
    loadPipeline,
    localStream,
    videoResolution,
    mode,
  });

  // Wire circular ref: download completion → handleStartStream
  const handleStartStreamRef = useRef<
    ((overridePipelineId?: PipelineId) => Promise<boolean>) | null
  >(null);
  handleStartStreamRef.current = streamLifecycle.handleStartStream;

  // Download handlers (needs handleStartStreamRef)
  const downloadHandlers = useDownloadHandlers({
    handleStartStreamRef,
    timelinePlayPauseRef,
    timelineRef,
  });

  return {
    ...settingsBridge,
    ...pipelineHandlers,
    ...downloadHandlers,
    ...streamLifecycle,
  };
}
