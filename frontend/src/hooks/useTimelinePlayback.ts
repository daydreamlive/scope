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
  const wasPausedByVideoRef = useRef<boolean>(false);
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
    console.log(
      "Starting playback, current time:",
      currentTime,
      "prompts:",
      prompts.length,
      "hasLiveBlocks:",
      prompts.some(p => p.isLive)
    );
    setIsPlaying(true);
    startTimeRef.current = performance.now() - currentTime * 1000;
    lastTimeRef.current = performance.now();

    const updateTime = () => {
      const now = performance.now();
      const elapsed = (now - startTimeRef.current) / 1000;

      console.log(
        "updateTime called, elapsed:",
        elapsed,
        "currentTime:",
        currentTime
      );
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

      // Check if there's a live block (recording in progress)
      const hasLiveBlock = currentPrompts.some(p => p.isLive);

      console.log("End check:", {
        elapsed,
        maxEndTime,
        hasLiveBlock,
        promptsCount: currentPrompts.length,
      });

      // If we've reached the end and there are prompts, pause automatically
      // BUT NOT if there's a live block (recording in progress)
      if (elapsed >= maxEndTime && currentPrompts.length > 0 && !hasLiveBlock) {
        console.log("Pausing at end of timeline");
        pausePlayback();
        return;
      }

      // Continue animation frame loop
      animationFrameRef.current = requestAnimationFrame(updateTime);
    };

    animationFrameRef.current = requestAnimationFrame(updateTime);
  }, [currentTime, prompts, pausePlayback]);

  // Pause timeline when video is paused (including during recording)
  useEffect(() => {
    console.log("Video pause effect:", {
      isVideoPaused: options?.isVideoPaused,
      isPlaying,
      hasLiveBlocks: promptsRef.current.some(p => p.isLive),
    });
    if (options?.isVideoPaused && isPlaying) {
      console.log("Pausing timeline due to video pause");
      wasPausedByVideoRef.current = true;
      pausePlayback();
    }
  }, [options?.isVideoPaused, isPlaying, pausePlayback]);

  // Resume timeline when video resumes (if it was playing before video pause)
  useEffect(() => {
    if (!options?.isVideoPaused && !isPlaying && wasPausedByVideoRef.current) {
      console.log("Resuming timeline due to video resume");
      wasPausedByVideoRef.current = false;
      startPlayback();
    }
  }, [options?.isVideoPaused, isPlaying, startPlayback]);

  const togglePlayback = useCallback(() => {
    console.log("togglePlayback called, isPlaying:", isPlaying);
    // Reset video pause tracking when user manually toggles
    wasPausedByVideoRef.current = false;

    if (isPlaying) {
      console.log("Pausing playback");
      pausePlayback();
    } else {
      console.log("Starting playback");
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
