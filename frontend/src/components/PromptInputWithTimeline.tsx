import React, { useState, useCallback, useEffect } from "react";

import { PromptTimeline, type TimelinePrompt } from "./PromptTimeline";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import type { PromptItem } from "../lib/api";
import type { SettingsState } from "../types";
import { generateRandomColor } from "../utils/promptColors";
import { submitTimelinePrompt } from "../utils/timelinePromptSubmission";

interface PromptInputWithTimelineProps {
  className?: string;
  currentPrompt: string;
  currentPromptItems?: PromptItem[];
  onPromptSubmit?: (prompt: string) => void;
  onPromptItemsSubmit?: (
    prompts: PromptItem[],
    transitionSteps?: number,
    temporalInterpolationMethod?: "linear" | "slerp"
  ) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  isVideoPaused?: boolean;
  timelineRef?: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
  } | null>;
  selectedPrompt?: TimelinePrompt | null;
  onPromptEdit?: (prompt: TimelinePrompt | null) => void;
  onLiveStateChange?: (isLive: boolean) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  onDisconnect?: () => void;
  onStartStream?: () => Promise<boolean> | void;
  onVideoPlayPauseToggle?: () => void;
  isCollapsed?: boolean;
  onCollapseToggle?: (collapsed: boolean) => void;
  externalSelectedPromptId?: string | null;
  settings?: SettingsState;
  onSettingsImport?: (settings: Partial<SettingsState>) => void;
  onPlayPauseRef?: React.RefObject<(() => Promise<void>) | null>;
  onVideoPlayingCallbackRef?: React.RefObject<(() => void) | null>;
  onResetCache?: () => void;
  onTimelinePromptsChange?: (prompts: TimelinePrompt[]) => void;
  onTimelineCurrentTimeChange?: (currentTime: number) => void;
  onTimelinePlayingChange?: (isPlaying: boolean) => void;
  isLoading?: boolean;
  transitionSteps?: number;
  temporalInterpolationMethod?: "linear" | "slerp";
  videoScaleMode?: "fit" | "native";
  onVideoScaleModeToggle?: () => void;
  isDownloading?: boolean;
  onSaveGeneration?: () => void;
}

export function PromptInputWithTimeline({
  className = "",
  currentPrompt,
  currentPromptItems = [],
  onPromptSubmit,
  onPromptItemsSubmit,
  disabled = false,
  isStreaming = false,
  isVideoPaused = false,
  timelineRef,
  selectedPrompt: _selectedPrompt = null,
  onPromptEdit,
  onLiveStateChange,
  onLivePromptSubmit,
  onDisconnect,
  onStartStream,
  onVideoPlayPauseToggle,
  isCollapsed = false,
  onCollapseToggle,
  externalSelectedPromptId = null,
  settings,
  onSettingsImport,
  onPlayPauseRef,
  onVideoPlayingCallbackRef,
  onResetCache,
  onTimelinePromptsChange,
  onTimelineCurrentTimeChange,
  onTimelinePlayingChange,
  isLoading = false,
  transitionSteps,
  temporalInterpolationMethod,
  videoScaleMode,
  onVideoScaleModeToggle,
  isDownloading = false,
  onSaveGeneration,
}: PromptInputWithTimelineProps) {
  const [isLive, setIsLive] = useState(false);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [hasStartedPlayback, setHasStartedPlayback] = useState(false);
  const [scrollToTimeFn, setScrollToTimeFn] = useState<
    ((time: number) => void) | null
  >(null);

  // Sync external selected prompt ID with internal state
  useEffect(() => {
    if (externalSelectedPromptId !== undefined) {
      setSelectedPromptId(externalSelectedPromptId);
    }
  }, [externalSelectedPromptId]);

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
    onPromptsChange: onTimelinePromptsChange,
    onCurrentTimeChange: onTimelineCurrentTimeChange,
    onPlayingChange: onTimelinePlayingChange,
  });

  // Compute actual playing state - timeline is playing AND video is not paused
  const isActuallyPlaying = isPlaying && !isVideoPaused;

  // Complete live prompt and reset to beginning
  const completeLivePrompt = useCallback(() => {
    // Check if there's actually a live prompt in the timeline
    const hasLivePrompt =
      prompts.length > 0 && prompts[prompts.length - 1].isLive;

    if (!hasLivePrompt) return;

    setIsLive(false);
    onLiveStateChange?.(false);

    setPrompts(prevPrompts => {
      if (prevPrompts.length === 0) return prevPrompts;

      const lastPrompt = prevPrompts[prevPrompts.length - 1];
      if (!lastPrompt.isLive) return prevPrompts;

      return [
        ...prevPrompts.slice(0, -1),
        {
          ...lastPrompt,
          endTime: currentTime,
          isLive: false,
          color: generateRandomColor(),
        },
      ];
    });
  }, [prompts, onLiveStateChange, currentTime, setPrompts]);

  // Reset to first prompt
  const resetToFirstPrompt = useCallback(() => {
    const firstPrompt = prompts.find(p => !p.isLive);

    if (firstPrompt) {
      submitTimelinePrompt(firstPrompt, {
        onPromptSubmit,
        onPromptItemsSubmit,
      });
    }
  }, [prompts, onPromptSubmit, onPromptItemsSubmit]);

  // Enhanced rewind handler
  const handleRewind = useCallback(() => {
    onResetCache?.();
    completeLivePrompt();
    updateCurrentTime(0);
    resetToFirstPrompt();
    scrollToTimeFn?.(0); // Scroll timeline back to beginning

    if (isActuallyPlaying) {
      pausePlayback();
      updateCurrentTime(0);
      setTimeout(() => startPlayback(), 10);
    }
  }, [
    onResetCache,
    completeLivePrompt,
    updateCurrentTime,
    resetToFirstPrompt,
    scrollToTimeFn,
    isActuallyPlaying,
    pausePlayback,
    startPlayback,
  ]);

  // Enhanced disconnect handler
  const handleEnhancedDisconnect = useCallback(() => {
    onDisconnect?.();

    if (isActuallyPlaying) {
      togglePlayback();
    }

    completeLivePrompt();
    updateCurrentTime(0);
    resetToFirstPrompt();
    scrollToTimeFn?.(0); // Scroll timeline back to beginning
  }, [
    onDisconnect,
    isActuallyPlaying,
    togglePlayback,
    completeLivePrompt,
    updateCurrentTime,
    resetToFirstPrompt,
    scrollToTimeFn,
  ]);

  // Reset hasStartedPlayback when stream stops
  React.useEffect(() => {
    if (!isStreaming) {
      setHasStartedPlayback(false);
    }
  }, [isStreaming]);

  const buildLivePromptFromCurrent = useCallback(
    (start: number, end: number): TimelinePrompt => {
      const basePrompt = {
        id: `live-${Date.now()}`,
        startTime: start,
        endTime: end,
        isLive: true,
        transitionSteps,
        temporalInterpolationMethod,
      };

      if (currentPromptItems?.length > 0) {
        return {
          ...basePrompt,
          text: currentPromptItems.map(p => p.text).join(", "),
          prompts: currentPromptItems.map(p => ({
            text: p.text,
            weight: p.weight,
          })),
        };
      }

      return {
        ...basePrompt,
        text: currentPrompt || "Live...",
      };
    },
    [
      currentPromptItems,
      currentPrompt,
      transitionSteps,
      temporalInterpolationMethod,
    ]
  );

  // Initialize stream if needed
  const initializeStream = useCallback(async (): Promise<boolean> => {
    if (!isStreaming && onStartStream) {
      const result = await onStartStream();
      const started = result === true; // Treat undefined/void as false
      if (started) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return true;
      }
      return false;
    }
    return isStreaming; // Already streaming
  }, [isStreaming, onStartStream]);

  // Check if at end of timeline
  const isAtTimelineEnd = useCallback(() => {
    const lastPrompt = prompts[prompts.length - 1];
    return !lastPrompt || currentTime >= lastPrompt.endTime;
  }, [prompts, currentTime]);

  // Deselect current prompt
  const deselectPrompt = useCallback(() => {
    if (selectedPromptId) {
      setSelectedPromptId(null);
      onPromptEdit?.(null);
    }
  }, [selectedPromptId, onPromptEdit]);

  // Handle starting playback
  const handleStartPlayback = useCallback(async () => {
    const streamStarted = await initializeStream();

    // Only proceed if stream successfully started
    if (!streamStarted) {
      return;
    }

    deselectPrompt();

    const isAtEnd = isAtTimelineEnd();

    if (isAtEnd) {
      setIsLive(true);
      onLiveStateChange?.(true);

      // Only create a new live prompt if there are no prompts at all in the timeline
      if (prompts.length === 0) {
        const streamStartedAgain = await initializeStream();
        if (streamStartedAgain) {
          const livePrompt = buildLivePromptFromCurrent(
            currentTime,
            currentTime
          );
          setPrompts(prevPrompts => [...prevPrompts, livePrompt]);
        }
      }
    }

    // Set callback to start playback when video actually starts playing
    if (onVideoPlayingCallbackRef) {
      onVideoPlayingCallbackRef.current = () => {
        togglePlayback();
      };
      // Unpause video immediately so it can start playing and fire the 'playing' event
      if (isVideoPaused) {
        onVideoPlayPauseToggle?.();
      }
    } else {
      // Fallback to old behavior if ref not provided
      setTimeout(() => {
        togglePlayback();
        if (isVideoPaused) {
          onVideoPlayPauseToggle?.();
        }
      }, 0);
    }

    if (!hasStartedPlayback) {
      setHasStartedPlayback(true);
    }
  }, [
    initializeStream,
    deselectPrompt,
    isAtTimelineEnd,
    onLiveStateChange,
    prompts,
    buildLivePromptFromCurrent,
    currentTime,
    setPrompts,
    togglePlayback,
    isVideoPaused,
    onVideoPlayPauseToggle,
    hasStartedPlayback,
    onVideoPlayingCallbackRef,
  ]);

  // Handle pausing playback
  const handlePausePlayback = useCallback(() => {
    togglePlayback();
    if (!isVideoPaused) {
      onVideoPlayPauseToggle?.();
    }
  }, [togglePlayback, isVideoPaused, onVideoPlayPauseToggle]);

  // Custom play/pause handler
  const handlePlayPause = useCallback(async () => {
    if (!isActuallyPlaying) {
      await handleStartPlayback();
    } else {
      handlePausePlayback();
    }
  }, [isActuallyPlaying, handleStartPlayback, handlePausePlayback]);

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

      // Check if the new prompt is the same as the current live prompt
      // Compare both text and weights for prompt blending
      console.log("handleLivePromptSubmit", promptItems);
      const newPromptText = promptItems.map(p => p.text).join(", ");
      const newPromptWeights = promptItems.map(p => p.weight);
      const currentLivePrompt = prompts.find(p => p.isLive);

      if (currentLivePrompt && currentLivePrompt.text === newPromptText) {
        // Also check if weights are the same for prompt blending
        const currentWeights =
          currentLivePrompt.prompts?.map(p => p.weight) || [];
        const weightsMatch =
          newPromptWeights.length === currentWeights.length &&
          newPromptWeights.every(
            (weight, index) =>
              Math.abs(weight - (currentWeights[index] || 0)) < 0.001
          );

        if (weightsMatch) {
          // Don't add duplicate prompt (same text and weights)
          return;
        }
      }

      setPrompts(prevPrompts => {
        let updatedPrompts = prevPrompts;

        // Only check the last prompt since only it can be live
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
      transitionSteps,
      temporalInterpolationMethod,
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

  // Simple timeline reset function
  const resetTimelineCompletely = useCallback(() => {
    // Reset all timeline state
    setPrompts([]);
    updateCurrentTime(0);

    if (isPlaying) {
      pausePlayback();
    }

    // Reset live state
    if (isLive) {
      setIsLive(false);
      onLiveStateChange?.(false);
    }

    // Reset selection state
    if (selectedPromptId !== null) {
      setSelectedPromptId(null);
    }

    // Reset playback state
    resetPlayback();

    // Notify parent components of state changes
    onTimelinePromptsChange?.([]);
    onTimelineCurrentTimeChange?.(0);
    onTimelinePlayingChange?.(false);
  }, [
    setPrompts,
    updateCurrentTime,
    isPlaying,
    pausePlayback,
    isLive,
    onLiveStateChange,
    selectedPromptId,
    setSelectedPromptId,
    resetPlayback,
    onTimelinePromptsChange,
    onTimelineCurrentTimeChange,
    onTimelinePlayingChange,
  ]);

  // Note: Live box end time updates are handled by useTimelinePlayback hook
  // to avoid conflicts and ensure proper synchronization

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
      />
    </div>
  );
}
