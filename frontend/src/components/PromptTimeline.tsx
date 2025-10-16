import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import {
  Play,
  Pause,
  Plus,
  Edit2,
  Trash2,
  Download,
  Upload,
  ZoomIn,
  ZoomOut,
  Square,
  ArrowUp,
} from "lucide-react";
import { Input } from "./ui/input";

export interface TimelinePrompt {
  id: string;
  text: string;
  startTime: number; // in seconds
  endTime: number; // in seconds
}

interface PromptTimelineProps {
  className?: string;
  prompts: TimelinePrompt[];
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  disabled?: boolean;
  isPlaying?: boolean;
  currentTime?: number; // in seconds
  onPlayPause?: () => void;
  onTimeChange?: (time: number) => void;
  isRecording?: boolean;
  onRecordingToggle?: () => void;
  onPromptSubmit?: (prompt: string) => void;
  initialPrompt?: string;
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
  isRecording = false,
  onRecordingToggle,
  onPromptSubmit,
  initialPrompt = "",
}: PromptTimelineProps) {
  const [editingPrompt, setEditingPrompt] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [draggingPrompt, setDraggingPrompt] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState(0);
  const [recordingPrompt, setRecordingPrompt] = useState("");
  const timelineRef = useRef<HTMLDivElement>(null);

  // Initialize recording prompt with initial prompt when recording starts
  useEffect(() => {
    if (isRecording && initialPrompt) {
      setRecordingPrompt(initialPrompt);
    }
  }, [isRecording, initialPrompt]);
  const [timelineWidth, setTimelineWidth] = useState(800);
  const [visibleStartTime, setVisibleStartTime] = useState(0);
  const [visibleEndTime, setVisibleEndTime] = useState(20); // Changed from 40 to 20
  const [zoomLevel, setZoomLevel] = useState(1); // 1 = 20s, 2 = 10s, 0.5 = 40s
  const basePixelsPerSecond = 20; // Base pixels per second
  const pixelsPerSecond = basePixelsPerSecond * zoomLevel; // Scaled pixels per second

  // Calculate visible time range based on zoom level and timeline width
  const visibleTimeRange = timelineWidth / pixelsPerSecond;

  // Update visible end time when zoom level or timeline width changes
  useEffect(() => {
    setVisibleEndTime(visibleStartTime + visibleTimeRange);
  }, [visibleStartTime, visibleTimeRange]);

  // Auto-scroll timeline during recording to follow the red line
  useEffect(() => {
    if (isRecording && currentTime > visibleEndTime - visibleTimeRange * 0.2) {
      // When the red line gets close to the right edge, scroll forward
      setVisibleStartTime(currentTime - visibleTimeRange * 0.8);
    } else if (
      isRecording &&
      currentTime < visibleStartTime + visibleTimeRange * 0.2
    ) {
      // When the red line gets close to the left edge, scroll backward
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.2));
    }
  }, [
    isRecording,
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

  const timeToPosition = useCallback(
    (time: number) => {
      return (time - visibleStartTime) * pixelsPerSecond;
    },
    [visibleStartTime, pixelsPerSecond]
  );

  const positionToTime = useCallback(
    (position: number) => {
      return visibleStartTime + position / pixelsPerSecond;
    },
    [visibleStartTime, pixelsPerSecond]
  );

  const updatePrompt = useCallback(
    (id: string, updates: Partial<TimelinePrompt>) => {
      onPromptsChange(
        prompts.map(prompt =>
          prompt.id === id ? { ...prompt, ...updates } : prompt
        )
      );
    },
    [prompts, onPromptsChange]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, prompt: TimelinePrompt) => {
      if (editingPrompt === prompt.id) return;

      e.preventDefault();
      setDraggingPrompt(prompt.id);

      const rect = timelineRef.current?.getBoundingClientRect();
      if (rect) {
        const clickX = e.clientX - rect.left;
        const promptStartX = timeToPosition(prompt.startTime);
        setDragOffset(clickX - promptStartX);
      }
    },
    [editingPrompt, timeToPosition]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!draggingPrompt || !timelineRef.current) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - dragOffset;
      const newStartTime = Math.max(0, positionToTime(mouseX));
      const currentPrompt = prompts.find(p => p.id === draggingPrompt);
      const duration = currentPrompt
        ? currentPrompt.endTime - currentPrompt.startTime
        : 10;

      updatePrompt(draggingPrompt, {
        startTime: newStartTime,
        endTime: newStartTime + duration,
      });

      // Auto-scroll if dragging near edges
      const scrollThreshold = 50;
      const scrollAmount = visibleTimeRange * 0.5; // Scroll by half the visible range
      if (mouseX < scrollThreshold && visibleStartTime > 0) {
        setVisibleStartTime(Math.max(0, visibleStartTime - scrollAmount));
      } else if (mouseX > timelineWidth - scrollThreshold) {
        setVisibleStartTime(visibleStartTime + scrollAmount);
      }
    },
    [
      draggingPrompt,
      dragOffset,
      positionToTime,
      prompts,
      updatePrompt,
      timelineWidth,
      visibleStartTime,
      visibleTimeRange,
    ]
  );

  const handleMouseUp = useCallback(() => {
    setDraggingPrompt(null);
    setDragOffset(0);
  }, []);

  const handleTimelineClick = useCallback(
    (e: React.MouseEvent) => {
      if (!timelineRef.current || isRecording) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const newTime = positionToTime(clickX);

      // Update current time via callback
      if (onTimeChange) {
        onTimeChange(Math.max(0, newTime));
      }
    },
    [positionToTime, onTimeChange, isRecording]
  );

  const handleExport = useCallback(() => {
    const timelineData = {
      prompts: prompts,
      version: "1.0",
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
  }, [prompts]);

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
            onPromptsChange(timelineData.prompts);
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
    [onPromptsChange]
  );

  // Add global mouse event listeners for dragging
  useEffect(() => {
    if (draggingPrompt) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [draggingPrompt, handleMouseMove, handleMouseUp]);

  const addPrompt = useCallback(() => {
    // Find the latest end time among existing prompts
    const latestEndTime =
      prompts.length > 0 ? Math.max(...prompts.map(p => p.endTime)) : -10;

    const newPrompt: TimelinePrompt = {
      id: Date.now().toString(),
      text: "New prompt",
      startTime: latestEndTime + 10, // Add 10 seconds after the last prompt (or 0s for first)
      endTime: latestEndTime + 20, // 10 second duration
    };
    onPromptsChange([...prompts, newPrompt]);
  }, [prompts, onPromptsChange]);

  const deletePrompt = useCallback(
    (id: string) => {
      onPromptsChange(prompts.filter(prompt => prompt.id !== id));
    },
    [prompts, onPromptsChange]
  );

  const startEditing = useCallback((prompt: TimelinePrompt) => {
    setEditingPrompt(prompt.id);
    setEditingText(prompt.text);
  }, []);

  const saveEditing = useCallback(() => {
    if (editingPrompt) {
      updatePrompt(editingPrompt, { text: editingText });
      setEditingPrompt(null);
      setEditingText("");
    }
  }, [editingPrompt, editingText, updatePrompt]);

  const cancelEditing = useCallback(() => {
    setEditingPrompt(null);
    setEditingText("");
  }, []);

  // Zoom functions
  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 2, 4)); // Max zoom 4x
  }, []);

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 2, 0.25)); // Min zoom 0.25x
  }, []);

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        saveEditing();
      } else if (e.key === "Escape") {
        cancelEditing();
      }
    },
    [saveEditing, cancelEditing]
  );

  const handleRecordingPromptSubmit = useCallback(() => {
    if (!recordingPrompt.trim()) return;

    // Add prompt to timeline at current time
    const newPrompt: TimelinePrompt = {
      id: Date.now().toString(),
      text: recordingPrompt.trim(),
      startTime: currentTime,
      endTime: currentTime + 10, // 10 second duration
    };

    onPromptsChange([...prompts, newPrompt]);

    // Also call the external prompt submit handler if provided
    if (onPromptSubmit) {
      onPromptSubmit(recordingPrompt.trim());
    }
  }, [recordingPrompt, currentTime, prompts, onPromptsChange, onPromptSubmit]);

  const handleRecordingKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        handleRecordingPromptSubmit();
      }
    },
    [handleRecordingPromptSubmit]
  );

  return (
    <Card className={`${className}`}>
      <CardContent className="p-4">
        {/* Timeline Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Button
              onClick={onPlayPause}
              disabled={disabled || isRecording}
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
              onClick={onRecordingToggle}
              disabled={disabled}
              size="sm"
              variant={isRecording ? "destructive" : "outline"}
              className={isRecording ? "animate-pulse" : ""}
            >
              {isRecording ? (
                <Square className="h-4 w-4" />
              ) : (
                <div className="h-4 w-4 rounded-full bg-red-500" />
              )}
              {isRecording ? "Stop" : "Record"}
            </Button>
            <span className="text-sm text-muted-foreground">
              {Math.floor(currentTime / 60)}:
              {Math.round(currentTime % 60)
                .toString()
                .padStart(2, "0")}
            </span>
            <div className="flex items-center gap-1 ml-4">
              <Button
                onClick={() => {
                  setVisibleStartTime(
                    Math.max(0, visibleStartTime - visibleTimeRange)
                  );
                }}
                disabled={visibleStartTime <= 0}
                size="sm"
                variant="outline"
                className="text-xs px-2"
              >
                ←
              </Button>
              <span className="text-xs text-muted-foreground px-2">
                {Math.round(visibleStartTime)}s - {Math.round(visibleEndTime)}s
              </span>
              <Button
                onClick={() => {
                  setVisibleStartTime(visibleStartTime + visibleTimeRange);
                }}
                size="sm"
                variant="outline"
                className="text-xs px-2"
              >
                →
              </Button>
            </div>
            <div className="flex items-center gap-1 ml-2">
              <Button
                onClick={zoomOut}
                disabled={zoomLevel <= 0.25}
                size="sm"
                variant="outline"
                className="text-xs px-2"
                title="Zoom Out"
              >
                <ZoomOut className="h-3 w-3" />
              </Button>
              <span className="text-xs text-muted-foreground px-1">
                {zoomLevel}x
              </span>
              <Button
                onClick={zoomIn}
                disabled={zoomLevel >= 4}
                size="sm"
                variant="outline"
                className="text-xs px-2"
                title="Zoom In"
              >
                <ZoomIn className="h-3 w-3" />
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={handleExport}
              disabled={disabled || prompts.length === 0 || isRecording}
              size="sm"
              variant="outline"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <div className="relative">
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={disabled || isRecording}
              />
              <Button
                size="sm"
                variant="outline"
                disabled={disabled || isRecording}
              >
                <Upload className="h-4 w-4 mr-1" />
                Import
              </Button>
            </div>
            <Button
              onClick={addPrompt}
              disabled={disabled || isRecording}
              size="sm"
              variant="outline"
            >
              <Plus className="h-4 w-4 mr-1" />
              Add Prompt
            </Button>
          </div>
        </div>

        {/* Recording Prompt Input */}
        {isRecording && (
          <div className="mb-4">
            <div className="flex items-center bg-card border border-border rounded-full px-4 py-3 gap-3">
              <Input
                placeholder="Enter prompt to add to timeline..."
                value={recordingPrompt}
                onChange={e => setRecordingPrompt(e.target.value)}
                onKeyPress={handleRecordingKeyPress}
                className="flex-1 bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0"
              />
              <Button
                onClick={handleRecordingPromptSubmit}
                disabled={!recordingPrompt.trim()}
                size="sm"
                className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Timeline */}
        <div className="relative overflow-hidden w-full" ref={timelineRef}>
          {/* Time markers */}
          <div className="relative mb-1 w-full" style={{ height: "30px" }}>
            {Array.from(
              {
                length: Math.ceil((visibleEndTime - visibleStartTime) / 10) + 1,
              },
              (_, i) => {
                const time = Math.round(visibleStartTime + i * 10); // Round to integer
                const position = timeToPosition(time);
                return (
                  <div
                    key={i}
                    className="absolute top-0 flex items-center justify-center"
                    style={{
                      left: time === 0 ? position + 10 : position,
                      transform: "translateX(-50%)",
                    }}
                  >
                    <span className="text-gray-400 text-xs">{time}s</span>
                  </div>
                );
              }
            )}
          </div>

          {/* Timeline track */}
          <div
            className="relative h-16 bg-muted rounded-lg border overflow-hidden cursor-pointer w-full"
            onClick={handleTimelineClick}
          >
            {/* Current time cursor */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-red-500 z-30 shadow-lg"
              style={{
                left: Math.max(
                  0,
                  Math.min(timelineWidth, timeToPosition(currentTime))
                ),
                display: "block",
              }}
            />
            {/* Debug info */}
            <div className="absolute top-0 right-0 text-xs text-white bg-black/80 px-2 py-1 rounded">
              Time: {Math.round(currentTime)}s
            </div>

            {/* Prompt start markers */}
            {prompts
              .filter(
                prompt =>
                  prompt.startTime >= visibleStartTime &&
                  prompt.startTime <= visibleEndTime
              )
              .map(prompt => (
                <div
                  key={`marker-${prompt.id}`}
                  className="absolute top-0 bottom-0 w-px bg-red-300/50 z-10"
                  style={{
                    left: Math.max(
                      0,
                      Math.min(timelineWidth, timeToPosition(prompt.startTime))
                    ),
                  }}
                />
              ))}

            {/* Prompt blocks */}
            {prompts
              .filter(
                prompt =>
                  prompt.endTime >= visibleStartTime &&
                  prompt.startTime <= visibleEndTime
              )
              .map(prompt => (
                <div
                  key={prompt.id}
                  className={`absolute top-2 bottom-2 bg-primary/20 border border-primary/40 rounded px-2 py-1 transition-colors ${
                    draggingPrompt === prompt.id
                      ? "cursor-grabbing bg-primary/30 shadow-lg"
                      : "cursor-grab hover:bg-primary/30"
                  }`}
                  style={{
                    left: Math.max(0, timeToPosition(prompt.startTime)),
                    width: Math.min(
                      timelineWidth -
                        Math.max(0, timeToPosition(prompt.startTime)),
                      timeToPosition(prompt.endTime) -
                        Math.max(0, timeToPosition(prompt.startTime))
                    ),
                  }}
                  onMouseDown={e => handleMouseDown(e, prompt)}
                >
                  <div className="flex items-center justify-between h-full">
                    <div className="flex-1 min-w-0">
                      {editingPrompt === prompt.id ? (
                        <Input
                          value={editingText}
                          onChange={e => setEditingText(e.target.value)}
                          onKeyDown={handleKeyPress}
                          onBlur={saveEditing}
                          className="h-6 text-xs bg-background text-foreground"
                          autoFocus
                        />
                      ) : (
                        <span className="text-xs text-primary-foreground font-medium truncate block">
                          {prompt.text}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-4 w-4 p-0 hover:bg-primary/20 text-primary-foreground"
                        onClick={e => {
                          e.stopPropagation();
                          startEditing(prompt);
                        }}
                      >
                        <Edit2 className="h-3 w-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-4 w-4 p-0 hover:bg-destructive/20 text-primary-foreground"
                        onClick={e => {
                          e.stopPropagation();
                          deletePrompt(prompt.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
