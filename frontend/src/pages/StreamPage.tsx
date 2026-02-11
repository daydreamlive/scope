import { useEffect, useRef } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { PromptInputWithTimeline } from "../components/PromptInputWithTimeline";
import { DownloadDialog } from "../components/DownloadDialog";
import type { TimelinePrompt } from "../components/PromptTimeline";
import { StatusBar } from "../components/StatusBar";
import { useUnifiedWebRTC } from "../hooks/useUnifiedWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useCloudContext } from "../lib/cloudContext";
import { useStreamActions } from "../hooks/useStreamActions";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import {
  StreamProvider,
  type StreamContextValue,
} from "../contexts/StreamContext";
import type { PromptItem } from "../lib/api";

export function StreamPage() {
  const { isCloudMode: isDirectCloudMode, isReady: isCloudReady } =
    useCloudContext();

  // Store state
  const {
    setPromptItems,
    setTemporalInterpolationMethod,
    setTransitionSteps,
    shouldReinitializeVideo,
    setCustomVideoResolution,
    selectedTimelinePrompt,
    setSelectedTimelinePrompt,
    isCloudConnecting,
    setExternalSelectedPromptId,
    setOpenSettingsTab,
    isDownloading,
  } = useAppStore(
    useShallow(s => ({
      setPromptItems: s.setPromptItems,
      setTemporalInterpolationMethod: s.setTemporalInterpolationMethod,
      setTransitionSteps: s.setTransitionSteps,
      shouldReinitializeVideo: s.shouldReinitializeVideo,
      setCustomVideoResolution: s.setCustomVideoResolution,
      selectedTimelinePrompt: s.selectedTimelinePrompt,
      setSelectedTimelinePrompt: s.setSelectedTimelinePrompt,
      isCloudConnecting: s.isCloudConnecting,
      setExternalSelectedPromptId: s.setExternalSelectedPromptId,
      setOpenSettingsTab: s.setOpenSettingsTab,
      isDownloading: s.isDownloading,
    }))
  );

  // Combined cloud mode
  const isCloudMode = isDirectCloudMode;

  useEffect(() => {
    if (isDirectCloudMode) {
      console.log("[StreamPage] Cloud mode enabled, ready:", isCloudReady);
    }
  }, [isDirectCloudMode, isCloudReady]);

  // Fetch available pipelines dynamically
  const { pipelines } = usePipelinesContext();

  // Stream state hook for settings management
  const { settings, updateSettings, getDefaults, spoutAvailable } =
    useStreamState();

  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
    pipelineInfo,
  } = usePipeline();

  // WebRTC
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
  } = useUnifiedWebRTC();

  const isLoading =
    isDownloading || isPipelineLoading || isConnecting || isCloudConnecting;

  const webrtcStats = useWebRTCStats({ peerConnectionRef, isStreaming });

  // Video source
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
    onCustomVideoResolution: resolution => {
      setCustomVideoResolution(resolution);
      updateSettings({
        resolution: { height: resolution.height, width: resolution.width },
      });
    },
  });

  // Refs for timeline
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
  const timelinePlayPauseRef = useRef<(() => Promise<void>) | null>(null);
  const onVideoPlayingCallbackRef = useRef<(() => void) | null>(null);

  // All handlers via useStreamActions
  const actions = useStreamActions({
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
  });

  // Open account tab after auth events
  useEffect(() => {
    const handleAuthEvent = () => setOpenSettingsTab("account");
    window.addEventListener("daydream-auth-success", handleAuthEvent);
    window.addEventListener("daydream-auth-error", handleAuthEvent);
    return () => {
      window.removeEventListener("daydream-auth-success", handleAuthEvent);
      window.removeEventListener("daydream-auth-error", handleAuthEvent);
    };
  }, [setOpenSettingsTab]);

  // ESC to exit timeline Edit mode
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selectedTimelinePrompt) {
        setSelectedTimelinePrompt(null);
        setExternalSelectedPromptId(null);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [
    selectedTimelinePrompt,
    setSelectedTimelinePrompt,
    setExternalSelectedPromptId,
  ]);

  // Update temporal interpolation defaults when pipeline changes
  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline) {
      const defaultMethod =
        pipeline.defaultTemporalInterpolationMethod || "slerp";
      const pipelineDefaultSteps =
        pipeline.defaultTemporalInterpolationSteps ?? 4;
      const modeDefaults = getDefaults(settings.pipelineId, settings.inputMode);
      const defaultSteps =
        modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps;

      setTemporalInterpolationMethod(defaultMethod);
      setTransitionSteps(defaultSteps);

      if (pipeline.supportsPrompts === false) {
        setPromptItems([{ text: "", weight: 1.0 }]);
      }
    }
  }, [
    settings.pipelineId,
    pipelines,
    settings.inputMode,
    getDefaults,
    setTemporalInterpolationMethod,
    setTransitionSteps,
    setPromptItems,
  ]);

  const streamContextValue: StreamContextValue = {
    sendParameterUpdate,
    isStreaming,
    isConnecting: isConnecting || isCloudConnecting,
    stopStream,
    remoteStream,
    actions,
    pipelineInfo,
    getDefaults,
    supportsNoiseControls: actions.supportsNoiseControls,
    spoutAvailable,
    updateSettings,
    isCloudMode,
    isLoading,
    isPipelineLoading,
  };

  return (
    <StreamProvider value={streamContextValue}>
      <div className="h-screen flex flex-col bg-background">
        <Header />

        <div className="flex-1 flex gap-4 px-4 pb-4 min-h-0 overflow-hidden">
          {/* Left Panel */}
          <div className="w-1/5">
            <InputAndControlsPanel
              className="h-full"
              localStream={localStream}
              isInitializing={isInitializing}
              error={videoSourceError}
              mode={mode}
              onVideoFileUpload={handleVideoFileUpload}
            />
          </div>

          {/* Center Panel */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 min-h-0">
              <VideoOutput
                className="h-full"
                pipelineError={pipelineError}
                onPlayPauseToggle={() => {
                  if (timelinePlayPauseRef.current) {
                    timelinePlayPauseRef.current();
                  }
                }}
                onStartStream={() => {
                  if (timelinePlayPauseRef.current) {
                    timelinePlayPauseRef.current();
                  }
                }}
                onVideoPlaying={() => {
                  if (onVideoPlayingCallbackRef.current) {
                    onVideoPlayingCallbackRef.current();
                    onVideoPlayingCallbackRef.current = null;
                  }
                }}
              />
            </div>
            <div className="flex-shrink-0 mt-2">
              <PromptInputWithTimeline
                timelineRef={timelineRef}
                onPlayPauseRef={timelinePlayPauseRef}
                onVideoPlayingCallbackRef={onVideoPlayingCallbackRef}
              />
            </div>
          </div>

          {/* Right Panel */}
          <div className="w-1/5 flex flex-col gap-3">
            <SettingsPanel className="flex-1 min-h-0 overflow-auto" />
          </div>
        </div>

        <StatusBar fps={webrtcStats.fps} bitrate={webrtcStats.bitrate} />

        <DownloadDialog />
      </div>
    </StreamProvider>
  );
}
