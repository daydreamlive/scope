import React, { useState, useCallback, useEffect } from "react";

import { PromptTimeline, type TimelinePrompt } from "./PromptTimeline";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import type { PromptItem } from "../lib/api";
import { generateRandomColor } from "../utils/promptColors";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { useStreamContext } from "../contexts/StreamContext";
import { usePromptSubmission } from "../hooks/usePromptSubmission";
import { usePlaybackControl } from "../hooks/usePlaybackControl";
import { useTimelineResets } from "../hooks/useTimelineResets";

interface PromptInputWithTimelineProps {
  className?: string;
  timelineRef?: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
  } | null>;
  onPlayPauseRef?: React.RefObject<(() => Promise<void>) | null>;
  onVideoPlayingCallbackRef?: React.RefObject<(() => void) | null>;
}

export function PromptInputWithTimeline({
  className = "",
  timelineRef,
  onPlayPauseRef,
  onVideoPlayingCallbackRef,
}: PromptInputWithTimelineProps) {
  const {
    settings,
    currentPromptItems,
    isCollapsed,
    onCollapseToggle,
    externalSelectedPromptId,
    videoScaleMode,
    setVideoScaleMode,
    isRecording,
    setIsRecording,
    isDownloading,
    showDownloadDialog,
    storeSetIsLive,
    setTimelinePrompts,
    setTimelineCurrentTime,
    setIsTimelinePlaying,
  } = useAppStore(
    useShallow(s => ({
      settings: s.settings,
      currentPromptItems: s.promptItems,
      isCollapsed: s.isTimelineCollapsed,
      onCollapseToggle: s.setIsTimelineCollapsed,
      externalSelectedPromptId: s.externalSelectedPromptId,
      videoScaleMode: s.videoScaleMode,
      setVideoScaleMode: s.setVideoScaleMode,
      isRecording: s.isRecording,
      setIsRecording: s.setIsRecording,
      isDownloading: s.isDownloading,
      showDownloadDialog: s.showDownloadDialog,
      storeSetIsLive: s.setIsLive,
      setTimelinePrompts: s.setTimelinePrompts,
      setTimelineCurrentTime: s.setTimelineCurrentTime,
      setIsTimelinePlaying: s.setIsTimelinePlaying,
    }))
  );

  const {
    actions,
    isStreaming,
    isConnecting,
    isLoading,
    isPipelineLoading,
    updateSettings,
  } = useStreamContext();

  // Derive values
  const currentPrompt = currentPromptItems[0]?.text || "";
  const isVideoPaused = settings.paused ?? false;
  const disabled = isPipelineLoading || isConnecting || showDownloadDialog;

  // Simple toggle handlers
  const onVideoScaleModeToggle = useCallback(
    () => setVideoScaleMode(videoScaleMode === "fit" ? "native" : "fit"),
    [videoScaleMode, setVideoScaleMode]
  );
  const onRecordingToggle = useCallback(
    () => setIsRecording(!isRecording),
    [isRecording, setIsRecording]
  );
  const onLiveStateChange = useCallback(
    (live: boolean) => storeSetIsLive(live),
    [storeSetIsLive]
  );

  // Hook: prompt submission
  const { onPromptSubmit, onPromptItemsSubmit } = usePromptSubmission();

  // Alias action callbacks
  const onPromptEdit = actions.handleTimelinePromptEdit;
  const onLivePromptSubmit = actions.handleLivePromptSubmit;
  const onSettingsImport = updateSettings;
  const onSaveGeneration = actions.handleSaveGeneration;

  // Local state
  const [isLive, setIsLive] = useState(false);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [scrollToTimeFn, setScrollToTimeFn] = useState<
    ((time: number) => void) | null
  >(null);

  // Sync external selected prompt ID with internal state
  useEffect(() => {
    if (externalSelectedPromptId !== undefined) {
      setSelectedPromptId(externalSelectedPromptId);
    }
  }, [externalSelectedPromptId]);

  // Timeline playback
  const {
    prompts,
    setPrompts,
    isPlaying,
    currentTime,
    updateCurrentTime,
    togglePlayback,
    resetPlayback,
    startPlayback,
    pausePlayback,
  } = useTimelinePlayback({
    onPromptChange: onPromptSubmit,
    onPromptItemsChange: onPromptItemsSubmit,
    isStreaming,
    isVideoPaused,
    onPromptsChange: setTimelinePrompts,
    onCurrentTimeChange: setTimelineCurrentTime,
    onPlayingChange: setIsTimelinePlaying,
  });

  const isActuallyPlaying = isPlaying && !isVideoPaused;

  // Hook: timeline resets
  const { handleRewind, handleEnhancedDisconnect, resetTimelineCompletely } =
    useTimelineResets({
      prompts,
      setPrompts,
      currentTime,
      updateCurrentTime,
      isPlaying,
      isActuallyPlaying,
      pausePlayback,
      startPlayback,
      togglePlayback,
      resetPlayback,
      scrollToTimeFn,
      isLive,
      setIsLive,
      onLiveStateChange,
      selectedPromptId,
      setSelectedPromptId,
      onPromptSubmit,
      onPromptItemsSubmit,
    });

  // Hook: playback control
  const { handlePlayPause } = usePlaybackControl({
    prompts,
    setPrompts,
    currentTime,
    togglePlayback,
    isActuallyPlaying,
    isVideoPaused,
    selectedPromptId,
    setSelectedPromptId,
    onLiveStateChange,
    onVideoPlayingCallbackRef,
  });

  // Expose current timeline prompt to parent
  const getCurrentTimelinePrompt = React.useCallback(() => {
    const activePrompt = prompts.find(
      prompt => currentTime >= prompt.startTime && currentTime <= prompt.endTime
    );
    return activePrompt?.text || "";
  }, [prompts, currentTime]);

  // Handle prompt selection
  const handlePromptSelect = React.useCallback((promptId: string | null) => {
    setSelectedPromptId(promptId);
  }, []);

  // Handle prompt editing
  const handlePromptEdit = React.useCallback(
    (prompt: TimelinePrompt | null) => {
      onPromptEdit?.(prompt);
    },
    [onPromptEdit]
  );

  // Handle live prompt submission
  const handleLivePromptSubmit = useCallback(
    (promptItems: PromptItem[]) => {
      if (!promptItems.length || !promptItems.some(p => p.text.trim())) {
        return;
      }

      console.log("handleLivePromptSubmit", promptItems);
      const newPromptText = promptItems.map(p => p.text).join(", ");
      const newPromptWeights = promptItems.map(p => p.weight);
      const currentLivePrompt = prompts.find(p => p.isLive);

      if (currentLivePrompt && currentLivePrompt.text === newPromptText) {
        const currentWeights =
          currentLivePrompt.prompts?.map(p => p.weight) || [];
        const weightsMatch =
          newPromptWeights.length === currentWeights.length &&
          newPromptWeights.every(
            (weight, index) =>
              Math.abs(weight - (currentWeights[index] || 0)) < 0.001
          );

        if (weightsMatch) {
          return;
        }
      }

      const transitionSteps = useAppStore.getState().transitionSteps;
      const temporalInterpolationMethod =
        useAppStore.getState().temporalInterpolationMethod;

      setPrompts(prevPrompts => {
        let updatedPrompts = prevPrompts;

        if (prevPrompts.length > 0) {
          const lastPrompt = prevPrompts[prevPrompts.length - 1];
          if (lastPrompt.isLive) {
            updatedPrompts = [
              ...prevPrompts.slice(0, -1),
              {
                ...lastPrompt,
                endTime: currentTime,
                isLive: false,
                color: generateRandomColor(),
              },
            ];
          }
        }

        const lastPrompt = updatedPrompts[updatedPrompts.length - 1];
        const maxEndTime = lastPrompt ? lastPrompt.endTime : 0;
        const isAtEnd = currentTime >= maxEndTime;
        const isPausedInMiddle = !isActuallyPlaying && !isAtEnd;
        const startTime = isPausedInMiddle ? maxEndTime : currentTime;

        const newLivePrompt: TimelinePrompt = {
          id: `live-${Date.now()}`,
          text: promptItems.map(p => p.text).join(", "),
          startTime,
          endTime: startTime,
          isLive: true,
          prompts: promptItems.map(p => ({ text: p.text, weight: p.weight })),
          transitionSteps,
          temporalInterpolationMethod,
        };

        return [...updatedPrompts, newLivePrompt];
      });

      setIsLive(true);
      onLiveStateChange?.(true);
      scrollToTimeFn?.(currentTime);
    },
    [
      currentTime,
      setPrompts,
      isActuallyPlaying,
      onLiveStateChange,
      scrollToTimeFn,
      prompts,
    ]
  );

  // Handle prompt updates from the editor
  const handlePromptUpdate = useCallback(
    (updatedPrompt: TimelinePrompt) => {
      setPrompts(prevPrompts =>
        prevPrompts.map(p => (p.id === updatedPrompt.id ? updatedPrompt : p))
      );
    },
    [setPrompts]
  );

  // Expose timeline methods to parent
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
    submitLivePrompt: handleLivePromptSubmit,
    updatePrompt: handlePromptUpdate,
    clearTimeline: () => setPrompts([]),
    resetPlayhead: resetPlayback,
    resetTimelineCompletely,
    getPrompts: () => prompts,
    getCurrentTime: () => currentTime,
    getIsPlaying: () => isPlaying,
  }));

  // Expose play/pause handler to parent
  useEffect(() => {
    if (onPlayPauseRef) {
      onPlayPauseRef.current = handlePlayPause;
    }
  }, [handlePlayPause, onPlayPauseRef]);

  return (
    <div className={`space-y-3 ${className}`}>
      <PromptTimeline
        prompts={prompts}
        onPromptsChange={setPrompts}
        disabled={disabled}
        isPlaying={isActuallyPlaying}
        currentTime={currentTime}
        onPlayPause={handlePlayPause}
        onTimeChange={handleRewind}
        onReset={handleEnhancedDisconnect}
        onClear={resetTimelineCompletely}
        onPromptSubmit={onPromptSubmit}
        initialPrompt={currentPrompt}
        selectedPromptId={selectedPromptId}
        onPromptSelect={handlePromptSelect}
        onPromptEdit={handlePromptEdit}
        onLivePromptSubmit={onLivePromptSubmit}
        isCollapsed={isCollapsed}
        onCollapseToggle={onCollapseToggle}
        settings={settings}
        onSettingsImport={onSettingsImport}
        onScrollToTime={scrollFn => setScrollToTimeFn(() => scrollFn)}
        isStreaming={isStreaming}
        isLoading={isLoading}
        videoScaleMode={videoScaleMode}
        onVideoScaleModeToggle={onVideoScaleModeToggle}
        isDownloading={isDownloading}
        onSaveGeneration={onSaveGeneration}
        isRecording={isRecording}
        onRecordingToggle={onRecordingToggle}
      />
    </div>
  );
}
