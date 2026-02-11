import React, { useCallback, useState, useEffect } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { useStreamContext } from "../contexts/StreamContext";

interface UsePlaybackControlParams {
  prompts: TimelinePrompt[];
  setPrompts: React.Dispatch<React.SetStateAction<TimelinePrompt[]>>;
  currentTime: number;
  togglePlayback: () => void;
  isActuallyPlaying: boolean;
  isVideoPaused: boolean;
  selectedPromptId: string | null;
  setSelectedPromptId: (id: string | null) => void;
  onLiveStateChange: (live: boolean) => void;
  onVideoPlayingCallbackRef?: React.RefObject<(() => void) | null>;
}

/**
 * Handles playback orchestration — starting, pausing, and toggling playback,
 * initializing streams, and managing live prompts at timeline end.
 */
export function usePlaybackControl({
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
}: UsePlaybackControlParams) {
  const { currentPromptItems, transitionSteps, temporalInterpolationMethod } =
    useAppStore(
      useShallow(s => ({
        currentPromptItems: s.promptItems,
        transitionSteps: s.transitionSteps,
        temporalInterpolationMethod: s.temporalInterpolationMethod,
      }))
    );

  const { isStreaming, actions } = useStreamContext();

  const currentPrompt = currentPromptItems[0]?.text || "";
  const onStartStream = actions.handleStartStream;
  const onVideoPlayPauseToggle = actions.handlePlayPauseToggle;
  const onPromptEdit = actions.handleTimelinePromptEdit;

  const [hasStartedPlayback, setHasStartedPlayback] = useState(false);

  // Reset hasStartedPlayback when stream stops
  useEffect(() => {
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
      const started = result === true;
      if (started) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return true;
      }
      return false;
    }
    return isStreaming;
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
  }, [selectedPromptId, onPromptEdit, setSelectedPromptId]);

  // Handle starting playback
  const handleStartPlayback = useCallback(async () => {
    const streamStarted = await initializeStream();

    if (!streamStarted) {
      return;
    }

    deselectPrompt();

    const isAtEnd = isAtTimelineEnd();

    if (isAtEnd) {
      onLiveStateChange?.(true);

      // Only create a new live prompt if there are no prompts at all
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
      if (isVideoPaused) {
        onVideoPlayPauseToggle?.();
      }
    } else {
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

  return {
    handlePlayPause,
    handleStartPlayback,
    handlePausePlayback,
    buildLivePromptFromCurrent,
  };
}
