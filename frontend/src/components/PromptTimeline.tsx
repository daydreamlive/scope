import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
} from "react";

import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import {
  Play,
  Pause,
  Download,
  Upload,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronUp,
  ChevronDown,
  Trash2,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { ExportDialog } from "./ExportDialog";

import type { PromptItem } from "../lib/api";
import type { SettingsState } from "../types";
import { generateRandomColor } from "../utils/promptColors";

// Timeline constants
const BASE_PIXELS_PER_SECOND = 20;
const MIN_DURATION_SECONDS = 0.5;
const MAX_ZOOM_LEVEL = 4;
const MIN_ZOOM_LEVEL = 0.25;
const DEFAULT_VISIBLE_END_TIME = 20;

// Utility functions
const timeToPosition = (
  time: number,
  visibleStartTime: number,
  pixelsPerSecond: number
): number => {
  return (time - visibleStartTime) * pixelsPerSecond;
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

// Helper function to get adjacent colors for color generation
const getAdjacentColors = (
  prompts: TimelinePrompt[],
  currentIndex: number
): string[] => {
  const adjacentColors: string[] = [];
  if (currentIndex > 0 && prompts[currentIndex - 1].color) {
    adjacentColors.push(prompts[currentIndex - 1].color!);
  }
  if (currentIndex < prompts.length - 1 && prompts[currentIndex + 1].color) {
    adjacentColors.push(prompts[currentIndex + 1].color!);
  }
  return adjacentColors;
};

// Helper function to calculate prompt box position
const calculatePromptPosition = (
  prompt: TimelinePrompt,
  index: number,
  visiblePrompts: TimelinePrompt[],
  timeToPositionFn: (time: number) => number
): number => {
  let leftPosition = Math.max(0, timeToPositionFn(prompt.startTime));

  if (index > 0) {
    const previousPrompt = visiblePrompts[index - 1];
    const previousEndPosition = Math.max(
      0,
      timeToPositionFn(previousPrompt.endTime)
    );
    leftPosition = Math.max(leftPosition, previousEndPosition);
  }

  return leftPosition;
};

// Helper function to get prompt box styling
const getPromptBoxStyle = (
  prompt: TimelinePrompt,
  leftPosition: number,
  timelineWidth: number,
  timeToPositionFn: (time: number) => number,
  isSelected: boolean,
  isLivePrompt: boolean,
  boxColor: string
) => {
  return {
    left: leftPosition,
    top: "8px",
    bottom: "8px",
    width: Math.min(
      timelineWidth - leftPosition,
      timeToPositionFn(prompt.endTime) - leftPosition
    ),
    backgroundColor: isLivePrompt ? "#6B7280" : boxColor,
    borderColor: isLivePrompt ? "#9CA3AF" : boxColor,
    opacity: isSelected ? 1.0 : 0.7,
  };
};

export interface TimelinePrompt {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  prompts?: Array<{ text: string; weight: number }>;
  color?: string;
  isLive?: boolean;
  transitionSteps?: number;
  temporalInterpolationMethod?: "linear" | "slerp";
}

// Timeline reset state
const TIMELINE_RESET_STATE = {
  prompts: [],
  currentTime: 0,
  isPlaying: false,
  isLive: false,
  isCollapsed: false,
  selectedPromptId: null,
  externalSelectedPromptId: null,
  selectedTimelinePrompt: null,
  timelineWidth: 800,
  visibleStartTime: 0,
  visibleEndTime: 20,
  zoomLevel: 1,
};

interface PromptTimelineProps {
  className?: string;
  prompts: TimelinePrompt[];
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  disabled?: boolean;
  isPlaying?: boolean;
  currentTime?: number;
  onPlayPause?: () => void;
  onTimeChange?: (time: number) => void;
  onReset?: () => void;
  onClear?: () => void;
  onPromptSubmit?: (prompt: string) => void;
  initialPrompt?: string;
  selectedPromptId?: string | null;
  onPromptSelect?: (promptId: string | null) => void;
  onPromptEdit?: (prompt: TimelinePrompt | null) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isCollapsed?: boolean;
  onCollapseToggle?: (collapsed: boolean) => void;
  settings?: SettingsState;
  onSettingsImport?: (settings: Partial<SettingsState>) => void;
  onScrollToTime?: (scrollFn: (time: number) => void) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  videoScaleMode?: "fit" | "native";
  onVideoScaleModeToggle?: () => void;
  isDownloading?: boolean;
  onSaveGeneration?: () => void;
}

export function PromptTimeline({
  className = "",
  prompts,
  onPromptsChange,
  disabled = false,
  isPlaying = false,
  currentTime = 0,
  onPlayPause,
  onTimeChange,
  onReset,
  onClear,
  onPromptSubmit: _onPromptSubmit,
  initialPrompt: _initialPrompt,
  selectedPromptId = null,
  onPromptSelect,
  onPromptEdit,
  onLivePromptSubmit: _onLivePromptSubmit,
  isCollapsed = false,
  onCollapseToggle,
  settings,
  onSettingsImport,
  onScrollToTime,
  isStreaming = false,
  isLoading = false,
  videoScaleMode = "fit",
  onVideoScaleModeToggle,
  isDownloading = false,
  onSaveGeneration,
}: PromptTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [timelineWidth, setTimelineWidth] = useState(800);
  const [visibleStartTime, setVisibleStartTime] = useState(0);
  const [visibleEndTime, setVisibleEndTime] = useState(
    DEFAULT_VISIBLE_END_TIME
  );
  const [zoomLevel, setZoomLevel] = useState(1);

  // Check if live mode is active
  const isLive = useMemo(() => prompts.some(p => p.isLive), [prompts]);

  // Memoized filtered prompts for better performance
  const visiblePrompts = useMemo(() => {
    return prompts.filter(
      prompt =>
        prompt.startTime !== prompt.endTime && // Exclude 0-length prompt boxes
        prompt.endTime >= visibleStartTime &&
        prompt.startTime <= visibleEndTime
    );
  }, [prompts, visibleStartTime, visibleEndTime]);

  // Calculate timeline metrics
  const pixelsPerSecond = useMemo(
    () => BASE_PIXELS_PER_SECOND * zoomLevel,
    [zoomLevel]
  );
  const visibleTimeRange = useMemo(
    () => timelineWidth / pixelsPerSecond,
    [timelineWidth, pixelsPerSecond]
  );

  // Scroll timeline to show a specific time
  const scrollToTime = useCallback(
    (time: number) => {
      const targetVisibleStartTime = Math.max(0, time - visibleTimeRange * 0.5);
      setVisibleStartTime(targetVisibleStartTime);
    },
    [visibleTimeRange]
  );

  // Expose scroll function to parent
  useEffect(() => {
    if (onScrollToTime) {
      onScrollToTime(scrollToTime);
    }
  }, [onScrollToTime, scrollToTime]);

  // Reset timeline UI state to initial values
  const resetTimelineUI = useCallback(() => {
    // Reset UI state
    setTimelineWidth(TIMELINE_RESET_STATE.timelineWidth);
    setVisibleStartTime(TIMELINE_RESET_STATE.visibleStartTime);
    setVisibleEndTime(TIMELINE_RESET_STATE.visibleEndTime);
    setZoomLevel(TIMELINE_RESET_STATE.zoomLevel);
  }, []);

  // Update visible end time when zoom level or timeline width changes
  useEffect(() => {
    setVisibleEndTime(visibleStartTime + visibleTimeRange);
  }, [visibleStartTime, visibleTimeRange]);

  // Auto-scroll timeline during playback to follow the red line
  useEffect(() => {
    // Don't auto-scroll if user is manually dragging or not playing
    if (isDraggingRef.current || !isPlaying) return;

    if (currentTime > visibleEndTime - visibleTimeRange * 0.2) {
      // When the red line gets close to the right edge, scroll forward
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.8));
    } else if (currentTime < visibleStartTime + visibleTimeRange * 0.2) {
      // When the red line gets close to the left edge, scroll backward
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.2));
    }
  }, [
    isPlaying,
    currentTime,
    visibleEndTime,
    visibleStartTime,
    visibleTimeRange,
  ]);

  // Update timeline width when component mounts or resizes
  useEffect(() => {
    const updateWidth = () => {
      if (timelineRef.current) {
        setTimelineWidth(timelineRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // Resize state
  const resizeStateRef = useRef<{
    promptId: string;
    edge: "left" | "right";
    startClientX: number;
    startPrompt: TimelinePrompt;
    prevPrompt?: TimelinePrompt;
    nextPrompt?: TimelinePrompt;
  } | null>(null);

  const beginResize = useCallback(
    (
      e: React.MouseEvent,
      prompt: TimelinePrompt,
      edge: "left" | "right",
      prevPrompt?: TimelinePrompt,
      nextPrompt?: TimelinePrompt
    ) => {
      e.stopPropagation();
      // Only prevent resizing if the stream is actively playing OR if this is a live prompt box
      if (isPlaying || prompt.isLive) return;
      resizeStateRef.current = {
        promptId: prompt.id,
        edge,
        startClientX: e.clientX,
        startPrompt: { ...prompt },
        prevPrompt: prevPrompt ? { ...prevPrompt } : undefined,
        nextPrompt: nextPrompt ? { ...nextPrompt } : undefined,
      };
      document.body.style.cursor = "col-resize";
    },
    [isPlaying]
  );

  // Memoized time-to-position conversion
  const timeToPositionMemo = useCallback(
    (time: number) => timeToPosition(time, visibleStartTime, pixelsPerSecond),
    [visibleStartTime, pixelsPerSecond]
  );

  // Memoized current time cursor position
  const currentTimePosition = useMemo(() => {
    return Math.max(
      0,
      Math.min(
        timelineWidth,
        (currentTime - visibleStartTime) * pixelsPerSecond
      )
    );
  }, [currentTime, timelineWidth, visibleStartTime, pixelsPerSecond]);

  // Memoized time markers for better performance
  const timeMarkers = useMemo(() => {
    return Array.from(
      {
        length: Math.ceil((visibleEndTime - visibleStartTime) / 10) + 1,
      },
      (_, i) => {
        const time = Math.round(visibleStartTime + i * 10);
        const position = (time - visibleStartTime) * pixelsPerSecond;
        return { time, position, index: i };
      }
    );
  }, [visibleEndTime, visibleStartTime, pixelsPerSecond]);

  const handlePromptClick = useCallback(
    (e: React.MouseEvent, prompt: TimelinePrompt) => {
      e.stopPropagation();
      // Only allow clicking if not playing (paused/stopped) or if already selected
      // Live prompts can't be clicked
      if (prompt.isLive) return;
      if (isPlaying && selectedPromptId !== prompt.id) return;

      // Check if this prompt is already selected
      const isCurrentlySelected = selectedPromptId === prompt.id;

      if (onPromptSelect) {
        // If already selected, deselect by passing null; otherwise select this prompt
        onPromptSelect(isCurrentlySelected ? null : prompt.id);
      }
      if (onPromptEdit) {
        // If already selected, pass null to deselect; otherwise pass the prompt
        onPromptEdit(isCurrentlySelected ? null : prompt);
      }
    },
    [selectedPromptId, onPromptSelect, onPromptEdit, isPlaying]
  );

  // Handle timeline clicks to deselect prompts when clicking on empty areas
  const handleTimelineClick = useCallback(
    (_e: React.MouseEvent) => {
      // Only deselect if clicking on the timeline background (not on a prompt box)
      // The prompt boxes will handle their own clicks via handlePromptClick
      if (selectedPromptId && onPromptSelect) {
        onPromptSelect(null);
      }
      if (selectedPromptId && onPromptEdit) {
        onPromptEdit(null);
      }
    },
    [selectedPromptId, onPromptSelect, onPromptEdit]
  );

  const [showExportDialog, setShowExportDialog] = useState(false);

  const handleSaveTimeline = useCallback(() => {
    // Filter out 0-length prompt boxes and only include prompts array and timing
    const exportPrompts = prompts
      .filter(prompt => prompt.startTime !== prompt.endTime) // Exclude 0-length prompt boxes
      .map(prompt => {
        const { id, text, isLive, color, ...exportPrompt } = prompt;

        // Always include a prompts array - convert single text to prompts format if needed
        if (!exportPrompt.prompts && text) {
          exportPrompt.prompts = [{ text, weight: 100 }];
        }

        // Suppress unused variable warnings for intentionally excluded fields
        void id;
        void text;
        void isLive;
        void color;
        return exportPrompt;
      });

    const timelineData = {
      prompts: exportPrompts,
      settings: settings
        ? {
            pipelineId: settings.pipelineId,
            inputMode: settings.inputMode,
            resolution: settings.resolution,
            seed: settings.seed,
            denoisingSteps: settings.denoisingSteps,
            noiseScale: settings.noiseScale,
            noiseController: settings.noiseController,
            manageCache: settings.manageCache,
            quantization: settings.quantization,
            kvCacheAttentionBias: settings.kvCacheAttentionBias,
            loras: settings.loras,
            loraMergeStrategy: settings.loraMergeStrategy,
            // Exclude paused state as it's runtime-specific
          }
        : undefined,
      version: "2.1", // Updated version to indicate inputMode inclusion
      exportedAt: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(timelineData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `timeline-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [prompts, settings]);

  const handleExport = useCallback(() => {
    setShowExportDialog(true);
  }, []);

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const content = e.target?.result as string;
          const timelineData = JSON.parse(content);

          if (timelineData.prompts && Array.isArray(timelineData.prompts)) {
            // Reset timeline UI state to initial values before importing
            resetTimelineUI();

            // Assign default values for id, text, isLive, and color when importing
            const importedPrompts = timelineData.prompts.map(
              (prompt: Partial<TimelinePrompt>, index: number) => ({
                ...prompt,
                id: prompt.id || `imported-${Date.now()}-${index}`,
                text:
                  prompt.text ||
                  (prompt.prompts && prompt.prompts.length > 0
                    ? prompt.prompts
                        .map((p: { text: string; weight: number }) => p.text)
                        .join(", ")
                    : ""),
                isLive: prompt.isLive || false,
                color: prompt.color || generateRandomColor(),
              })
            );
            onPromptsChange(importedPrompts);

            // Adjust visible timeline range to show all imported prompts
            if (importedPrompts.length > 0) {
              const maxEndTime = Math.max(
                ...importedPrompts.map((p: TimelinePrompt) => p.endTime || 0)
              );
              // Set visible end time to show all prompts, with minimum of default visible range
              const newVisibleEndTime = Math.max(
                maxEndTime + 2, // Add 2 seconds buffer
                DEFAULT_VISIBLE_END_TIME
              );
              setVisibleStartTime(0);
              setVisibleEndTime(newVisibleEndTime);
            }

            // Import settings if available and callback is provided
            if (timelineData.settings && onSettingsImport) {
              onSettingsImport(timelineData.settings);
            }

            // Automatically move playhead to beginning after import
            onTimeChange?.(0);

            // Apply the first prompt from the imported timeline
            if (importedPrompts.length > 0 && _onPromptSubmit) {
              const firstPrompt = importedPrompts[0];
              _onPromptSubmit(firstPrompt.text);
            }
          } else {
            alert("Invalid timeline file format");
          }
        } catch (error) {
          alert("Error reading timeline file");
          console.error("Import error:", error);
        }
      };
      reader.readAsText(file);

      // Reset the input so the same file can be selected again
      event.target.value = "";
    },
    [
      onPromptsChange,
      onSettingsImport,
      resetTimelineUI,
      onTimeChange,
      _onPromptSubmit,
    ]
  );

  // Drag-to-pan state
  const isDraggingRef = useRef(false);
  const dragStartXRef = useRef(0);
  const dragStartVisibleStartRef = useRef(0);

  // Handle mouse down on the track to begin panning
  const handleTimelineMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!timelineRef.current) return;
      isDraggingRef.current = true;
      dragStartXRef.current = e.clientX;
      dragStartVisibleStartRef.current = visibleStartTime;
      // Change cursor to grabbing while dragging
      document.body.style.cursor = "grabbing";
    },
    [visibleStartTime]
  );

  // Global listeners to update panning and finish drag
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Resize has priority over panning
      if (resizeStateRef.current) {
        const state = resizeStateRef.current;
        const deltaX = e.clientX - state.startClientX;
        const deltaSeconds = deltaX / pixelsPerSecond;

        // Find the prompt index directly since prompts are in chronological order
        const index = prompts.findIndex(p => p.id === state.promptId);
        if (index === -1) return;

        const current = { ...state.startPrompt };
        const prev = index > 0 ? prompts[index - 1] : null;
        const next = index < prompts.length - 1 ? prompts[index + 1] : null;

        if (state.edge === "left") {
          let newStart = current.startTime + deltaSeconds;
          const leftBound = prev ? prev.startTime + MIN_DURATION_SECONDS : 0;
          const rightBound = current.endTime - MIN_DURATION_SECONDS;
          newStart = Math.max(leftBound, Math.min(newStart, rightBound));

          current.startTime = newStart;
          if (prev) {
            // Keep adjacency
            prev.endTime = newStart;
          }
        } else {
          let newEnd = current.endTime + deltaSeconds;
          const leftBound = current.startTime + MIN_DURATION_SECONDS;
          const rightBound = next
            ? next.endTime - MIN_DURATION_SECONDS
            : Number.POSITIVE_INFINITY;
          newEnd = Math.max(leftBound, Math.min(newEnd, rightBound));

          current.endTime = newEnd;
          if (next) {
            // Keep adjacency
            next.startTime = newEnd;
          }
        }

        const updated = prompts.map((p, i) => {
          if (i === index) return current;
          if (state.edge === "left" && prev && i === index - 1) return prev;
          if (state.edge === "right" && next && i === index + 1) return next;
          return p;
        });
        onPromptsChange(updated);
        return;
      }

      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - dragStartXRef.current;
      const deltaSeconds = -deltaX / pixelsPerSecond;
      const nextStart = Math.max(
        0,
        dragStartVisibleStartRef.current + deltaSeconds
      );
      setVisibleStartTime(nextStart);
    };

    const handleMouseUp = () => {
      if (resizeStateRef.current) {
        resizeStateRef.current = null;
        document.body.style.cursor = "";
        return;
      }
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        document.body.style.cursor = "";
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [pixelsPerSecond, prompts, onPromptsChange]);

  // Zoom functions
  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 2, MAX_ZOOM_LEVEL));
  }, []);

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 2, MIN_ZOOM_LEVEL));
  }, []);

  return (
    <Card className={`${className}`}>
      <CardContent className={`p-4 ${isCollapsed ? "py-2" : ""}`}>
        {/* Timeline Header */}
        <div
          className={`flex items-center justify-between ${isCollapsed ? "mb-0" : "mb-4"}`}
        >
          <div className="flex items-center gap-2">
            <Button
              onClick={onPlayPause}
              disabled={disabled || isLoading || isDownloading}
              size="sm"
              variant="outline"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            <Button
              onClick={onReset}
              disabled={disabled || isLoading || isDownloading}
              size="sm"
              variant="outline"
              title="Reset timeline"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              onClick={onClear}
              disabled={
                disabled ||
                isPlaying ||
                isStreaming ||
                isLoading ||
                isDownloading
              }
              size="sm"
              variant="outline"
              title="Clear timeline"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {onVideoScaleModeToggle && (
              <Button
                onClick={onVideoScaleModeToggle}
                size="sm"
                variant="outline"
                disabled={!isStreaming}
                title={
                  videoScaleMode === "fit"
                    ? "Switch to native resolution"
                    : "Switch to fit to window"
                }
              >
                {videoScaleMode === "fit" ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
            )}
            <Button
              onClick={handleExport}
              disabled={disabled || isLoading || isDownloading}
              size="sm"
              variant="outline"
            >
              <Upload className="h-4 w-4 mr-1" />
              Export
            </Button>
            <ExportDialog
              open={showExportDialog}
              onClose={() => setShowExportDialog(false)}
              onSaveGeneration={() => {
                if (onSaveGeneration) {
                  onSaveGeneration();
                }
              }}
              onSaveTimeline={handleSaveTimeline}
            />
            <div className="relative">
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={disabled || isStreaming || isLoading || isDownloading}
              />
              <Button
                size="sm"
                variant="outline"
                disabled={disabled || isStreaming || isLoading || isDownloading}
              >
                <Download className="h-4 w-4 mr-1" />
                Import
              </Button>
            </div>
            <Button
              onClick={() => onCollapseToggle?.(!isCollapsed)}
              size="sm"
              variant="outline"
              disabled={isLoading || isDownloading}
              title={isCollapsed ? "Expand timeline" : "Collapse timeline"}
            >
              {isCollapsed ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronUp className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Timeline */}
        {!isCollapsed && (
          <div className="relative overflow-hidden w-full" ref={timelineRef}>
            {/* Time markers */}
            <div className="relative mb-1 w-full" style={{ height: "30px" }}>
              {timeMarkers.map(({ time, position, index }) => (
                <div
                  key={index}
                  className="absolute top-0 flex items-center justify-center"
                  style={{
                    left: time === 0 ? position + 10 : position,
                    transform: "translateX(-50%)",
                  }}
                >
                  <span className="text-gray-400 text-xs">
                    {formatTime(time)}
                  </span>
                </div>
              ))}
            </div>

            {/* Timeline track */}
            <div
              className="relative bg-muted rounded-lg border overflow-hidden cursor-grab w-full"
              style={{ height: "80px" }} // Compact height for timeline display
              onClick={handleTimelineClick}
              onMouseDown={handleTimelineMouseDown}
            >
              {/* Zoom controls positioned at bottom right */}
              <div className="absolute bottom-2 right-2 flex items-center gap-1 z-50">
                <Button
                  onClick={zoomOut}
                  disabled={zoomLevel <= MIN_ZOOM_LEVEL}
                  size="sm"
                  variant="outline"
                  className="text-xs px-2 h-6"
                  title="Zoom Out"
                >
                  <ZoomOut className="h-3 w-3" />
                </Button>
                <span className="text-xs text-muted-foreground px-1">
                  {zoomLevel}x
                </span>
                <Button
                  onClick={zoomIn}
                  disabled={zoomLevel >= MAX_ZOOM_LEVEL}
                  size="sm"
                  variant="outline"
                  className="text-xs px-2 h-6"
                  title="Zoom In"
                >
                  <ZoomIn className="h-3 w-3" />
                </Button>
              </div>
              {/* Current time cursor */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-red-500 z-30 shadow-lg"
                style={{
                  left: currentTimePosition,
                  display: "block",
                }}
              />
              {/* Removed debug time overlay */}

              {/* Prompt blocks */}
              {visiblePrompts.map((prompt, index) => {
                const isSelected = selectedPromptId === prompt.id;
                const isActive =
                  currentTime >= prompt.startTime &&
                  currentTime <= prompt.endTime;
                const isLiveActive = isLive && currentTime >= prompt.startTime;
                const isLivePrompt = prompt.isLive;

                // Use the prompt's color, only generate if it doesn't exist
                let boxColor = prompt.color;
                if (!boxColor) {
                  const adjacentColors = getAdjacentColors(
                    visiblePrompts,
                    index
                  );
                  boxColor = generateRandomColor(adjacentColors);

                  // Update the prompt with the new color to persist it
                  const updatedPrompt = { ...prompt, color: boxColor };
                  const updatedPrompts = prompts.map(p =>
                    p.id === prompt.id ? updatedPrompt : p
                  );
                  onPromptsChange(updatedPrompts);
                }

                // Calculate position - boxes should be adjacent with no gaps
                const leftPosition = calculatePromptPosition(
                  prompt,
                  index,
                  visiblePrompts,
                  timeToPositionMemo
                );

                const prevPrompt =
                  index > 0 ? visiblePrompts[index - 1] : undefined;
                const nextPrompt =
                  index < visiblePrompts.length - 1
                    ? visiblePrompts[index + 1]
                    : undefined;

                // Determine if this prompt box is editable
                const isEditable = !isPlaying || isSelected || isLivePrompt;

                return (
                  <div
                    key={prompt.id}
                    className={`absolute border rounded px-2 py-1 transition-colors ${
                      isEditable ? "cursor-pointer" : "cursor-default"
                    } ${
                      isSelected
                        ? "shadow-lg border-blue-500"
                        : isActive
                          ? "border-green-500"
                          : isLiveActive
                            ? "border-red-500"
                            : ""
                    }`}
                    style={getPromptBoxStyle(
                      prompt,
                      leftPosition,
                      timelineWidth,
                      timeToPositionMemo,
                      isSelected,
                      isLivePrompt || false,
                      boxColor
                    )}
                    onClick={e => handlePromptClick(e, prompt)}
                  >
                    {/* Resize handles */}
                    {!isPlaying && !isLivePrompt && (
                      <>
                        <div
                          className="absolute top-0 bottom-0 w-2 -left-1 z-40"
                          style={{ cursor: "col-resize" }}
                          onMouseDown={e =>
                            beginResize(
                              e,
                              prompt,
                              "left",
                              prevPrompt,
                              nextPrompt
                            )
                          }
                        />
                        <div
                          className="absolute top-0 bottom-0 w-2 -right-1 z-40"
                          style={{ cursor: "col-resize" }}
                          onMouseDown={e =>
                            beginResize(
                              e,
                              prompt,
                              "right",
                              prevPrompt,
                              nextPrompt
                            )
                          }
                        />
                      </>
                    )}
                    <div className="flex flex-col justify-center h-full">
                      <div className="flex-1 flex flex-col justify-center">
                        {prompt.prompts && prompt.prompts.length > 1 ? (
                          // Display blend prompts vertically
                          prompt.prompts.map((promptItem, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-white font-medium truncate"
                            >
                              {promptItem.text} ({promptItem.weight}%)
                            </div>
                          ))
                        ) : (
                          // Single prompt display
                          <span className="text-xs text-white font-medium truncate">
                            {prompt.text}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
