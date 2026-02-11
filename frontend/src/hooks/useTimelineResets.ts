import React, { useCallback } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";
import type { PromptItem } from "../lib/api";
import { generateRandomColor } from "../utils/promptColors";
import { submitTimelinePrompt } from "../utils/timelinePromptSubmission";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { useStreamContext } from "../contexts/StreamContext";

interface UseTimelineResetsParams {
  prompts: TimelinePrompt[];
  setPrompts: React.Dispatch<React.SetStateAction<TimelinePrompt[]>>;
  currentTime: number;
  updateCurrentTime: (time: number) => void;
  isPlaying: boolean;
  isActuallyPlaying: boolean;
  pausePlayback: () => void;
  startPlayback: () => void;
  togglePlayback: () => void;
  resetPlayback: () => void;
  scrollToTimeFn: ((time: number) => void) | null;
  isLive: boolean;
  setIsLive: (live: boolean) => void;
  onLiveStateChange: (live: boolean) => void;
  selectedPromptId: string | null;
  setSelectedPromptId: (id: string | null) => void;
  onPromptSubmit: (text: string) => void;
  onPromptItemsSubmit: (
    prompts: PromptItem[],
    blockTransitionSteps?: number,
    blockTemporalInterpolationMethod?: "linear" | "slerp"
  ) => void;
}

/**
 * Handles timeline reset operations — completing live prompts, rewinding,
 * disconnecting with cleanup, and fully resetting the timeline.
 */
export function useTimelineResets({
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
}: UseTimelineResetsParams) {
  const { setTimelinePrompts, setTimelineCurrentTime, setIsTimelinePlaying } =
    useAppStore(
      useShallow(s => ({
        setTimelinePrompts: s.setTimelinePrompts,
        setTimelineCurrentTime: s.setTimelineCurrentTime,
        setIsTimelinePlaying: s.setIsTimelinePlaying,
      }))
    );

  const { stopStream, actions } = useStreamContext();
  const onResetCache = actions.handleResetCache;

  // Complete live prompt and reset to beginning
  const completeLivePrompt = useCallback(() => {
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
  }, [prompts, onLiveStateChange, currentTime, setPrompts, setIsLive]);

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
    scrollToTimeFn?.(0);

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
    stopStream?.();

    if (isActuallyPlaying) {
      togglePlayback();
    }

    completeLivePrompt();
    updateCurrentTime(0);
    resetToFirstPrompt();
    scrollToTimeFn?.(0);
  }, [
    stopStream,
    isActuallyPlaying,
    togglePlayback,
    completeLivePrompt,
    updateCurrentTime,
    resetToFirstPrompt,
    scrollToTimeFn,
  ]);

  // Simple timeline reset function
  const resetTimelineCompletely = useCallback(() => {
    setPrompts([]);
    updateCurrentTime(0);

    if (isPlaying) {
      pausePlayback();
    }

    if (isLive) {
      setIsLive(false);
      onLiveStateChange?.(false);
    }

    if (selectedPromptId !== null) {
      setSelectedPromptId(null);
    }

    resetPlayback();

    setTimelinePrompts?.([]);
    setTimelineCurrentTime?.(0);
    setIsTimelinePlaying?.(false);
  }, [
    setPrompts,
    updateCurrentTime,
    isPlaying,
    pausePlayback,
    isLive,
    setIsLive,
    onLiveStateChange,
    selectedPromptId,
    setSelectedPromptId,
    resetPlayback,
    setTimelinePrompts,
    setTimelineCurrentTime,
    setIsTimelinePlaying,
  ]);

  return {
    completeLivePrompt,
    resetToFirstPrompt,
    handleRewind,
    handleEnhancedDisconnect,
    resetTimelineCompletely,
  };
}
