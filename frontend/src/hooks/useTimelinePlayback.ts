import { useState, useRef, useCallback, useEffect } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";

interface UseTimelinePlaybackOptions {
  onPromptChange?: (prompt: string) => void;
  isStreaming?: boolean;
  isVideoPaused?: boolean;
}

export function useTimelinePlayback(options?: UseTimelinePlaybackOptions) {
  const [prompts, setPrompts] = useState<TimelinePrompt[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const animationFrameRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const lastAppliedPromptIdRef = useRef<string | null>(null);
  const lastAppliedPromptTextRef = useRef<string | null>(null);
  const optionsRef = useRef(options);
  const promptsRef = useRef(prompts);

  // Update options ref when options change
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  // Update prompts ref when prompts change
  useEffect(() => {
    promptsRef.current = prompts;
  }, [prompts]);

  const pausePlayback = useCallback(() => {
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  const startPlayback = useCallback(() => {
    setIsPlaying(true);
    startTimeRef.current = performance.now() - currentTime * 1000;
    lastTimeRef.current = performance.now();

    const updateTime = () => {
      const now = performance.now();
      const elapsed = (now - startTimeRef.current) / 1000;

      setCurrentTime(elapsed);

      // Update live blocks to extend as time progresses
      const currentPrompts = promptsRef.current;
      const hasLiveBlocks = currentPrompts.some(p => p.isLive);
      if (hasLiveBlocks) {
        setPrompts(prevPrompts =>
          prevPrompts.map(p => (p.isLive ? { ...p, endTime: elapsed } : p))
        );
      }

      // Find active prompt and apply it only if it changed
      const activePrompt = currentPrompts.find(
        prompt => elapsed >= prompt.startTime && elapsed <= prompt.endTime
      );

      // Only send update if the active prompt block OR its text has changed
      if (
        activePrompt &&
        (activePrompt.id !== lastAppliedPromptIdRef.current ||
          activePrompt.text !== lastAppliedPromptTextRef.current)
      ) {
        if (optionsRef.current?.onPromptChange) {
          optionsRef.current.onPromptChange(activePrompt.text);
        }
        lastAppliedPromptIdRef.current = activePrompt.id;
        lastAppliedPromptTextRef.current = activePrompt.text;
      } else if (!activePrompt && lastAppliedPromptIdRef.current !== null) {
        // No active prompt, reset the last applied prompt
        lastAppliedPromptIdRef.current = null;
        lastAppliedPromptTextRef.current = null;
      }

      // Check if we've reached the end of all prompts
      const sortedPrompts = [...currentPrompts].sort(
        (a, b) => a.endTime - b.endTime
      );
      const lastPrompt = sortedPrompts[sortedPrompts.length - 1];
      const maxEndTime = lastPrompt ? lastPrompt.endTime : 0;

      // Check if there's a live block (live mode in progress)
      const hasLiveBlock = currentPrompts.some(p => p.isLive);

      // If we've reached the end and there are prompts, extend the last box as live
      // instead of stopping (unless there's already a live block)
      if (elapsed >= maxEndTime && currentPrompts.length > 0 && !hasLiveBlock) {
        // Find the last non-live prompt and make it live
        const sortedNonLivePrompts = [...currentPrompts]
          .filter(p => !p.isLive)
          .sort((a, b) => a.endTime - b.endTime);

        if (sortedNonLivePrompts.length > 0) {
          const lastNonLivePrompt =
            sortedNonLivePrompts[sortedNonLivePrompts.length - 1];

          // Make the last prompt live
          setPrompts(prevPrompts =>
            prevPrompts.map(p =>
              p.id === lastNonLivePrompt.id
                ? { ...p, isLive: true, endTime: elapsed }
                : p
            )
          );

          // Notify parent that live mode has started
          if (optionsRef.current?.onPromptChange) {
            optionsRef.current.onPromptChange(lastNonLivePrompt.text);
          }
        }

        // Continue playing instead of stopping
        animationFrameRef.current = requestAnimationFrame(updateTime);
        return;
      }

      // Continue animation frame loop
      animationFrameRef.current = requestAnimationFrame(updateTime);
    };

    animationFrameRef.current = requestAnimationFrame(updateTime);
  }, [currentTime]);

  // Note: Video pause/resume effects removed - now handled by unified play/pause handler

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }, [isPlaying, startPlayback, pausePlayback]);

  const resetPlayback = useCallback(() => {
    pausePlayback();
    setCurrentTime(0);
  }, [pausePlayback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const updateCurrentTime = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  return {
    prompts,
    setPrompts,
    isPlaying,
    currentTime,
    updateCurrentTime,
    togglePlayback,
    resetPlayback,
    startPlayback,
    pausePlayback,
  };
}
