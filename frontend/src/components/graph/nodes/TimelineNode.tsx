import {
  Handle,
  Position,
  useEdges,
  useNodes,
  useUpdateNodeInternals,
} from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef, useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getNumberFromNode } from "../utils/getValueFromNode";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodePillInput,
  collapsedHandleStyle,
} from "../ui";

type TimelineNodeType = Node<FlowNodeData, "timeline">;

const TRIGGER_COLOR = "#f59e0b";
const PLAYHEAD_COLOR = "#ef4444"; // red-500
const TRACK_HEIGHT = 40;
const RULER_HEIGHT = 16;
const HEADER_HEIGHT = 28;

/** Generate a short unique id for a trigger */
function generateTriggerId(): string {
  return "t_" + Math.random().toString(36).slice(2, 8);
}

export function TimelineNode({
  id,
  data,
  selected,
}: NodeProps<TimelineNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const updateNodeInternals = useUpdateNodeInternals();

  const duration = data.timelineDuration ?? 10;
  const triggers = data.timelineTriggers ?? [];
  const isPlaying = data.isPlaying ?? false;
  const loop = data.timelineLoop ?? false;
  const currentTime = data.timelineCurrentTime ?? 0;

  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const startTimeRef = useRef<number>(0);
  const offsetRef = useRef<number>(0);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const firedTriggersRef = useRef<Set<string>>(new Set());
  const lastPlayInputRef = useRef<number>(0);
  const trackRef = useRef<HTMLDivElement>(null);

  // ── Force React Flow to re-register handles when triggers change ──
  const triggerIdsKey = triggers.map(t => t.id).join(",");
  useEffect(() => {
    // Small delay to let DOM render the new Handle elements first
    const timer = setTimeout(() => {
      updateNodeInternals(id);
    }, 20);
    return () => clearTimeout(timer);
  }, [triggerIdsKey, id, updateNodeInternals]);

  // ── Play input: watch for rising edge from connected source ──
  useEffect(() => {
    const playHandleId = buildHandleId("param", "play");
    const playEdge = edges.find(
      e => e.target === id && e.targetHandle === playHandleId
    );
    if (!playEdge) return;

    const sourceNode = allNodes.find(n => n.id === playEdge.source);
    if (!sourceNode) return;

    const val = getNumberFromNode(sourceNode, playEdge.sourceHandle) ?? 0;
    const prev = lastPlayInputRef.current;
    lastPlayInputRef.current = val;

    // Rising edge detection
    if (prev <= 0 && val > 0 && !isPlaying) {
      handlePlay();
    } else if (prev > 0 && val <= 0 && isPlaying) {
      handlePause();
    }
  }); // intentionally no deps - runs every render to detect changes

  // ── Playback animation loop ──
  // Uses wall-clock timestamps (Date.now()) so the timeline can "catch up"
  // after being unmounted (e.g. entering a subgraph) and remounted.
  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
      return;
    }

    // If we have a stored wall-clock start, reconstruct elapsed time so
    // the playhead catches up to the real position after remount.
    const wallStart = data._timelineWallStart as number | undefined;
    const wallOffset = data._timelineWallOffset as number | undefined;
    if (wallStart && wallOffset !== undefined) {
      const wallElapsed = (Date.now() - wallStart) / 1000;
      let reconstructed = wallOffset + wallElapsed;
      if (loop && duration > 0) {
        reconstructed = reconstructed % duration;
      } else if (reconstructed > duration) {
        // Finished while unmounted
        updateData({
          timelineCurrentTime: duration,
          isPlaying: false,
          triggerValues: {},
          _timelineWallStart: undefined,
          _timelineWallOffset: undefined,
        });
        return;
      }
      startTimeRef.current = performance.now();
      offsetRef.current = reconstructed;
    } else {
      startTimeRef.current = performance.now();
      offsetRef.current = currentTime;
    }

    firedTriggersRef.current.clear();
    // Mark triggers already past the reconstructed offset as fired
    for (const trigger of triggers) {
      if (trigger.time <= offsetRef.current) {
        firedTriggersRef.current.add(trigger.id);
      }
    }

    const animate = () => {
      const elapsed = (performance.now() - startTimeRef.current) / 1000;
      let newTime = offsetRef.current + elapsed;

      if (newTime >= duration) {
        if (loop) {
          newTime = newTime % duration;
          startTimeRef.current = performance.now();
          offsetRef.current = 0;
          firedTriggersRef.current.clear();
          // Update wall-clock anchor for loop restart
          updateData({
            _timelineWallStart: Date.now(),
            _timelineWallOffset: 0,
          });
        } else {
          newTime = duration;
          updateData({
            timelineCurrentTime: duration,
            isPlaying: false,
            triggerValues: {},
            _timelineWallStart: undefined,
            _timelineWallOffset: undefined,
          });
          return;
        }
      }

      // Check triggers
      const newTriggerValues: Record<string, number> = {};
      let anyTriggered = false;
      for (const trigger of triggers) {
        if (
          trigger.time <= newTime &&
          !firedTriggersRef.current.has(trigger.id)
        ) {
          firedTriggersRef.current.add(trigger.id);
          newTriggerValues[trigger.id] = 1;
          anyTriggered = true;
        }
      }

      // Update node data
      const updatePayload: Partial<FlowNodeData> = {
        timelineCurrentTime: newTime,
      };
      if (anyTriggered) {
        updatePayload.triggerValues = {
          ...(data.triggerValues ?? {}),
          ...newTriggerValues,
        };
        const keysToReset = Object.keys(newTriggerValues);
        setTimeout(() => {
          const resetValues: Record<string, number> = {};
          for (const key of keysToReset) {
            resetValues[key] = 0;
          }
          updateData({ triggerValues: resetValues });
        }, 50);
      }
      updateData(updatePayload);

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, loop, duration, triggers.length]);

  const handlePlay = useCallback(() => {
    const startFrom = currentTime >= duration ? 0 : currentTime;
    updateData({
      timelineCurrentTime: startFrom,
      isPlaying: true,
      triggerValues: startFrom === 0 ? {} : undefined,
      _timelineWallStart: Date.now(),
      _timelineWallOffset: startFrom,
    });
  }, [currentTime, duration, updateData]);

  const handlePause = useCallback(() => {
    updateData({
      isPlaying: false,
      _timelineWallStart: undefined,
      _timelineWallOffset: undefined,
    });
  }, [updateData]);

  const handleStop = useCallback(() => {
    updateData({
      isPlaying: false,
      timelineCurrentTime: 0,
      triggerValues: {},
      _timelineWallStart: undefined,
      _timelineWallOffset: undefined,
    });
  }, [updateData]);

  const handleTogglePlay = useCallback(() => {
    if (isPlaying) {
      handlePause();
    } else {
      handlePlay();
    }
  }, [isPlaying, handlePause, handlePlay]);

  // ── Add trigger on double-click ──
  const handleTrackDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      if (!trackRef.current) return;
      e.stopPropagation();
      const rect = trackRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      const time = ratio * duration;

      const newTrigger = {
        id: generateTriggerId(),
        time: parseFloat(time.toFixed(2)),
      };
      updateData({
        timelineTriggers: [...triggers, newTrigger].sort(
          (a, b) => a.time - b.time
        ),
      });
    },
    [duration, triggers, updateData]
  );

  // ── Remove trigger ──
  const handleRemoveTrigger = useCallback(
    (triggerId: string) => {
      updateData({
        timelineTriggers: triggers.filter(t => t.id !== triggerId),
      });
    },
    [triggers, updateData]
  );

  // ── Drag playhead ──
  const handlePlayheadPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (!trackRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);

      const setTimeFromMouse = (clientX: number) => {
        if (!trackRef.current) return;
        const rect = trackRef.current.getBoundingClientRect();
        const ratio = Math.max(
          0,
          Math.min(1, (clientX - rect.left) / rect.width)
        );
        const time = ratio * duration;
        updateData({ timelineCurrentTime: parseFloat(time.toFixed(3)) });
      };

      setTimeFromMouse(e.clientX);

      const onMove = (ev: PointerEvent) => setTimeFromMouse(ev.clientX);
      const onUp = () => {
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [duration, updateData]
  );

  // ── Compute playhead position ──
  const playheadPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  // ── Format time display ──
  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = t % 60;
    return `${mins}:${secs.toFixed(1).padStart(4, "0")}`;
  };

  // ── Ruler tick marks ──
  const tickCount = Math.min(Math.max(Math.floor(duration), 2), 20);
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => i / tickCount);

  // Handle Y position for play input — center of the header
  const playHandleY = HEADER_HEIGHT / 2;

  return (
    <NodeCard
      selected={selected}
      collapsed={collapsed}
      minWidth={400}
      minHeight={100}
    >
      <NodeHeader
        title={data.customTitle || "Timeline"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        rightContent={
          !collapsed ? (
            <div className="flex items-center gap-1.5">
              {/* Duration */}
              <span className="text-[9px] text-[#8c8c8d]">Dur</span>
              <NodePillInput
                type="number"
                value={duration}
                onChange={v =>
                  updateData({ timelineDuration: Math.max(0.1, Number(v)) })
                }
                min={0.1}
                className="!w-[52px]"
              />
              {/* Loop */}
              <label className="flex items-center gap-0.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={loop}
                  onChange={e =>
                    updateData({ timelineLoop: e.target.checked })
                  }
                  className="w-3 h-3 rounded accent-amber-500"
                />
                <span className="text-[9px] text-[#8c8c8d]">Loop</span>
              </label>
              {/* Stop */}
              <button
                onClick={e => {
                  e.stopPropagation();
                  handleStop();
                }}
                className="w-4 h-4 flex items-center justify-center text-[#888] hover:text-red-400 transition-colors text-[10px]"
                type="button"
                title="Stop"
              >
                ■
              </button>
              {/* Play / Pause */}
              <button
                onClick={e => {
                  e.stopPropagation();
                  handleTogglePlay();
                }}
                className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-amber-400 transition-colors"
                type="button"
                title={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? "⏸" : "▶"}
              </button>
            </div>
          ) : undefined
        }
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <NodeBody>
          {/* Time display bar */}
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] text-[#8c8c8d] font-mono tabular-nums">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
            <span className="text-[8px] text-[#555]">
              {triggers.length} trigger{triggers.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Ruler */}
          <div
            className="relative w-full select-none"
            style={{ height: RULER_HEIGHT }}
          >
            {ticks.map((pct, i) => (
              <div
                key={i}
                className="absolute top-0 flex flex-col items-center"
                style={{
                  left: `${pct * 100}%`,
                  transform: "translateX(-50%)",
                }}
              >
                <div className="w-px h-2 bg-[#555]" />
                <span className="text-[7px] text-[#666] leading-none mt-px">
                  {(pct * duration).toFixed(pct === 0 || pct === 1 ? 0 : 1)}
                </span>
              </div>
            ))}
          </div>

          {/* Track area (double-click to add triggers, drag playhead) */}
          <div
            ref={trackRef}
            className="relative w-full rounded cursor-pointer"
            style={{
              height: TRACK_HEIGHT,
              background: "#1b1a1a",
              border: "1px solid rgba(119,119,119,0.15)",
            }}
            onDoubleClick={handleTrackDoubleClick}
            onPointerDown={handlePlayheadPointerDown}
          >
            {/* Trigger markers */}
            {triggers.map(trigger => {
              const pct =
                duration > 0 ? (trigger.time / duration) * 100 : 0;
              const isFired =
                (data.triggerValues?.[trigger.id] ?? 0) > 0;
              return (
                <div
                  key={trigger.id}
                  className="absolute top-0 bottom-0 flex flex-col items-center group/trigger"
                  style={{
                    left: `${pct}%`,
                    transform: "translateX(-50%)",
                    zIndex: 2,
                  }}
                >
                  {/* Vertical line */}
                  <div
                    className="w-px h-full"
                    style={{
                      backgroundColor: isFired ? "#fbbf24" : TRIGGER_COLOR,
                      opacity: isFired ? 1 : 0.5,
                      boxShadow: isFired ? "0 0 6px #fbbf24" : "none",
                    }}
                  />
                  {/* Marker dot */}
                  <div
                    className="absolute top-0 w-2 h-2 rounded-full -translate-y-1/2"
                    style={{
                      backgroundColor: isFired ? "#fbbf24" : TRIGGER_COLOR,
                      boxShadow: isFired
                        ? "0 0 8px #fbbf24"
                        : "0 0 3px rgba(0,0,0,0.5)",
                    }}
                  />
                  {/* Remove button (on hover) */}
                  <button
                    className="absolute -top-3 opacity-0 group-hover/trigger:opacity-100 w-3 h-3 rounded-full bg-red-500/80 text-white text-[7px] flex items-center justify-center leading-none hover:bg-red-500 transition-opacity"
                    onClick={e => {
                      e.stopPropagation();
                      e.preventDefault();
                      handleRemoveTrigger(trigger.id);
                    }}
                    onPointerDown={e => e.stopPropagation()}
                    type="button"
                    title={`Remove trigger at ${trigger.time.toFixed(1)}s`}
                  >
                    ×
                  </button>
                  {/* Label */}
                  <span className="absolute bottom-0 translate-y-full text-[7px] text-[#666] whitespace-nowrap">
                    {trigger.label || `${trigger.time.toFixed(1)}s`}
                  </span>
                </div>
              );
            })}

            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 pointer-events-none"
              style={{
                left: `${playheadPct}%`,
                backgroundColor: PLAYHEAD_COLOR,
                boxShadow: `0 0 4px ${PLAYHEAD_COLOR}`,
                zIndex: 3,
              }}
            >
              {/* Playhead triangle */}
              <div
                className="absolute -top-1 left-1/2 -translate-x-1/2"
                style={{
                  width: 0,
                  height: 0,
                  borderLeft: "4px solid transparent",
                  borderRight: "4px solid transparent",
                  borderTop: `5px solid ${PLAYHEAD_COLOR}`,
                }}
              />
            </div>

            {/* Hint text when empty */}
            {triggers.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-[9px] text-[#555] pointer-events-none">
                Double-click to add triggers
              </div>
            )}
          </div>
        </NodeBody>
      )}

      {/* ── Handles ── */}

      {/* Play input (left side, vertically centered on header) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "play")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : {
                top: playHandleY,
                left: 0,
                backgroundColor: "#38bdf8",
              }
        }
      />

      {/* Trigger outputs (top side) */}
      {triggers.map(trigger => {
        const pct = duration > 0 ? (trigger.time / duration) * 100 : 0;
        return (
          <Handle
            key={trigger.id}
            type="source"
            position={Position.Top}
            id={buildHandleId("param", `trigger_${trigger.id}`)}
            className="!w-2.5 !h-2.5 !border-0"
            style={
              collapsed
                ? {
                    top: 0,
                    left: "50%",
                    backgroundColor: "#9ca3af",
                    opacity: 0,
                  }
                : {
                    left: `${pct}%`,
                    top: 0,
                    backgroundColor: TRIGGER_COLOR,
                  }
            }
          />
        );
      })}
    </NodeCard>
  );
}
