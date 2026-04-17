import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NODE_TOKENS,
  NodePillInput,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_DEFAULT, COLOR_BOOLEAN } from "../nodeColors";

type SchedulerNodeType = Node<FlowNodeData, "scheduler">;

interface TriggerEntry {
  time: number;
  port_name: string;
}

const COLOR_TRIGGER = "#f97316";
const COLOR_FLOAT = COLOR_DEFAULT;

function triggersKey(entries: TriggerEntry[]): string {
  return entries.map(e => `${e.port_name}@${e.time}`).join("|");
}

export function SchedulerNode({
  id,
  data,
  selected,
}: NodeProps<SchedulerNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  // Read state from FlowNodeData
  const isPlaying = (data.schedulerIsPlaying as boolean) ?? false;
  const elapsed = (data.schedulerElapsed as number) ?? 0;

  const rawTriggers = data.schedulerTriggers as TriggerEntry[] | undefined;
  const triggers = useMemo(
    () => rawTriggers ?? [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rawTriggers ? triggersKey(rawTriggers) : ""]
  );
  const duration = (data.schedulerDuration as number) ?? 30;
  const loop = (data.schedulerLoop as boolean) ?? false;

  // Refs for RAF timer
  const startTimeRef = useRef<number>(0);
  const rafRef = useRef<number | undefined>(undefined);
  const firedRef = useRef<Set<string>>(new Set());
  const fireCountsRef = useRef<Record<string, number>>(
    (data.schedulerFireCounts as Record<string, number>) ?? {}
  );
  const tickRef = useRef<number>((data.schedulerTickCount as number) ?? 0);
  const lastElapsedEmitRef = useRef<number>(0);

  // Sync refs when data changes externally (e.g., graph load)
  useEffect(() => {
    fireCountsRef.current =
      (data.schedulerFireCounts as Record<string, number>) ?? {};
    tickRef.current = (data.schedulerTickCount as number) ?? 0;
  }, [data.schedulerFireCounts, data.schedulerTickCount]);

  // Watch external start trigger counter → toggle play
  const startCount = (data._schedulerStartCount as number) ?? 0;
  const prevStartCountRef = useRef(startCount);
  useEffect(() => {
    if (startCount > 0 && startCount !== prevStartCountRef.current) {
      prevStartCountRef.current = startCount;
      updateData({ schedulerIsPlaying: !isPlaying });
    }
  }, [startCount, isPlaying, updateData]);

  // Watch external reset trigger counter → reset timer
  const resetCount = (data._schedulerResetCount as number) ?? 0;
  const prevResetCountRef = useRef(resetCount);
  useEffect(() => {
    if (resetCount > 0 && resetCount !== prevResetCountRef.current) {
      prevResetCountRef.current = resetCount;
      // Reset all state
      fireCountsRef.current = {};
      tickRef.current = 0;
      firedRef.current.clear();
      startTimeRef.current = Date.now();
      updateData({
        schedulerElapsed: 0,
        schedulerFireCounts: {},
        schedulerTickCount: 0,
      });
    }
  }, [resetCount, updateData]);

  // RAF-based timer loop
  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }

    startTimeRef.current = Date.now() - elapsed * 1000;
    lastElapsedEmitRef.current = 0;
    // Rebuild fired set for already-passed triggers
    firedRef.current = new Set();
    for (const t of triggers) {
      if (t.time <= elapsed) {
        firedRef.current.add(`${t.port_name}@${t.time}`);
      }
    }

    const animate = () => {
      const now = Date.now();
      let currentElapsed = (now - startTimeRef.current) / 1000;

      // Check triggers
      const newFirings: string[] = [];
      for (const trigger of triggers) {
        const key = `${trigger.port_name}@${trigger.time}`;
        if (!firedRef.current.has(key) && trigger.time <= currentElapsed) {
          firedRef.current.add(key);
          newFirings.push(trigger.port_name);
        }
      }

      // Handle looping / auto-stop
      if (duration > 0 && currentElapsed >= duration) {
        if (loop) {
          startTimeRef.current = now;
          currentElapsed = 0;
          firedRef.current.clear();
        } else {
          // Auto-stop
          updateData({
            schedulerIsPlaying: false,
            schedulerElapsed: duration,
          });
          return; // don't schedule next RAF
        }
      }

      // Update fire counts and tick
      if (newFirings.length > 0) {
        const counts = { ...fireCountsRef.current };
        for (const port of newFirings) {
          counts[port] = (counts[port] ?? 0) + 1;
        }
        fireCountsRef.current = counts;
        tickRef.current += 1;
        updateData({
          schedulerFireCounts: counts,
          schedulerTickCount: tickRef.current,
        });
      }

      // Throttled elapsed update (~50ms)
      if (now - lastElapsedEmitRef.current >= 50) {
        updateData({
          schedulerElapsed: Math.round(currentElapsed * 1000) / 1000,
        });
        lastElapsedEmitRef.current = now;
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, triggersKey(triggers), duration, loop]);

  const handleTogglePlay = useCallback(() => {
    if (isPlaying) {
      updateData({ schedulerIsPlaying: false });
    } else {
      updateData({ schedulerIsPlaying: true });
    }
  }, [isPlaying, updateData]);

  const handleReset = useCallback(() => {
    const wasPlaying = isPlaying;
    updateData({
      schedulerIsPlaying: false,
      schedulerElapsed: 0,
      schedulerFireCounts: {},
      schedulerTickCount: 0,
    });
    fireCountsRef.current = {};
    tickRef.current = 0;
    firedRef.current.clear();
    if (wasPlaying) {
      // Restart after reset
      setTimeout(() => updateData({ schedulerIsPlaying: true }), 0);
    }
  }, [isPlaying, updateData]);

  // Trigger list management
  const addTrigger = useCallback(() => {
    const name = `trigger_${triggers.length + 1}`;
    const time = Math.min(duration, elapsed + 1);
    const next = [...triggers, { time, port_name: name }];
    updateData({ schedulerTriggers: next });
  }, [triggers, duration, elapsed, updateData]);

  const removeTrigger = useCallback(
    (idx: number) => {
      const next = triggers.filter((_, i) => i !== idx);
      updateData({ schedulerTriggers: next });
    },
    [triggers, updateData]
  );

  const updateTrigger = useCallback(
    (idx: number, field: "time" | "port_name", value: string | number) => {
      const next = triggers.map((t, i) =>
        i === idx ? { ...t, [field]: value } : t
      );
      updateData({ schedulerTriggers: next });
    },
    [triggers, updateData]
  );

  const updateDuration = useCallback(
    (v: number) => {
      updateData({ schedulerDuration: Math.max(1, v) });
    },
    [updateData]
  );

  const toggleLoop = useCallback(() => {
    updateData({ schedulerLoop: !loop });
  }, [loop, updateData]);

  // Dynamic output ports
  const dynamicPortNames = useMemo(() => {
    const staticNames = new Set(["tick", "elapsed", "is_playing"]);
    const seen = new Set<string>();
    return triggers
      .filter(t => {
        if (staticNames.has(t.port_name) || seen.has(t.port_name)) return false;
        seen.add(t.port_name);
        return true;
      })
      .map(t => t.port_name);
  }, [triggers]);

  const staticOutputs = ["tick", "elapsed", "is_playing"];
  const allOutputs = [...dynamicPortNames, ...staticOutputs];
  const inputNames = ["start", "reset"];

  const allRows = useMemo(
    () => [...inputNames, ...allOutputs],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dynamicPortNames.join(",")]
  );
  const { setRowRef, rowPositions } = useHandlePositions(allRows);

  // Flash animation for trigger ports
  const [flashingPorts, setFlashingPorts] = useState<Set<string>>(new Set());
  const flashTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map()
  );
  const lastCounters = useRef<Map<string, number>>(new Map());

  const fireCounts = (data.schedulerFireCounts as Record<string, number>) ?? {};
  const tickCount = (data.schedulerTickCount as number) ?? 0;

  useEffect(() => {
    const checkPorts = [
      ...dynamicPortNames.map(name => ({
        name,
        counter: fireCounts[name] ?? 0,
      })),
      { name: "tick", counter: tickCount },
    ];

    for (const { name, counter } of checkPorts) {
      const prev = lastCounters.current.get(name) ?? 0;
      if (counter > 0 && counter !== prev) {
        lastCounters.current.set(name, counter);
        setFlashingPorts(p => {
          if (p.has(name)) return p;
          const next = new Set(p);
          next.add(name);
          return next;
        });
        const existing = flashTimers.current.get(name);
        if (existing) clearTimeout(existing);
        flashTimers.current.set(
          name,
          setTimeout(() => {
            setFlashingPorts(p => {
              const next = new Set(p);
              next.delete(name);
              return next;
            });
            flashTimers.current.delete(name);
          }, 200)
        );
      }
    }
  }, [fireCounts, tickCount, dynamicPortNames]);

  // Flash edges connected to flashing output ports
  const { setEdges } = useReactFlow();
  const edgeFlashTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map()
  );

  useEffect(() => {
    // Find ports that just started flashing (compare with previous)
    const newlyFlashing = new Set<string>();
    for (const name of flashingPorts) {
      newlyFlashing.add(name);
    }
    if (newlyFlashing.size === 0) return;

    // Build set of handle IDs that are flashing
    const flashHandleIds = new Set<string>();
    for (const name of newlyFlashing) {
      flashHandleIds.add(buildHandleId("param", name));
    }

    // Set flashing=true on matching edges
    setEdges(edges =>
      edges.map(e => {
        if (
          e.source === id &&
          e.sourceHandle &&
          flashHandleIds.has(e.sourceHandle)
        ) {
          // Only set if not already flashing (avoid unnecessary updates)
          if (e.data?.flashing) return e;
          return { ...e, data: { ...e.data, flashing: true } };
        }
        return e;
      })
    );

    // Schedule clear for each flashing port
    for (const name of newlyFlashing) {
      const handleId = buildHandleId("param", name);
      const existing = edgeFlashTimers.current.get(name);
      if (existing) clearTimeout(existing);
      edgeFlashTimers.current.set(
        name,
        setTimeout(() => {
          setEdges(edges =>
            edges.map(e => {
              if (
                e.source === id &&
                e.sourceHandle === handleId &&
                e.data?.flashing
              ) {
                return { ...e, data: { ...e.data, flashing: false } };
              }
              return e;
            })
          );
          edgeFlashTimers.current.delete(name);
        }, 200)
      );
    }
  }, [flashingPorts, id, setEdges]);

  function outputColor(name: string): string {
    if (flashingPorts.has(name)) return "#ffffff";
    if (name === "elapsed") return COLOR_FLOAT;
    if (name === "is_playing") return COLOR_BOOLEAN;
    return COLOR_TRIGGER;
  }

  const nodeName = data.label ?? "Scheduler";

  return (
    <NodeCard
      selected={selected}
      collapsed={collapsed}
      minWidth={320}
      autoMinHeight={!collapsed}
    >
      <NodeHeader
        title={data.customTitle || nodeName}
        onTitleChange={t => updateData({ customTitle: t })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
        rightContent={
          !collapsed && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] font-mono tabular-nums text-[#666]">
                {elapsed.toFixed(1)}s
              </span>
              <button
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  isPlaying
                    ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                    : "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                }`}
                onClick={handleTogglePlay}
              >
                {isPlaying ? "\u25A0" : "\u25B6"}
              </button>
              <button
                className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors bg-[#333] text-[#999] hover:bg-[#444] hover:text-[#ccc]"
                onClick={handleReset}
              >
                {"\u21BA"}
              </button>
            </div>
          )
        }
      />

      {!collapsed && (
        <NodeBody withGap>
          {/* Transport row */}
          <div className="flex items-center gap-2 text-[10px]">
            <label className="flex items-center gap-1.5 text-[#8c8c8d] cursor-pointer select-none">
              <button
                type="button"
                role="checkbox"
                aria-checked={loop}
                onClick={toggleLoop}
                className={`w-3.5 h-3.5 rounded-[3px] border flex items-center justify-center transition-colors ${
                  loop
                    ? "bg-blue-500 border-blue-500"
                    : "bg-[#1a1a1a] border-[rgba(255,255,255,0.15)] hover:border-[rgba(255,255,255,0.3)]"
                }`}
              >
                {loop && (
                  <svg
                    className="w-2.5 h-2.5 text-white"
                    viewBox="0 0 12 12"
                    fill="none"
                  >
                    <path
                      d="M2.5 6L5 8.5L9.5 3.5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
              </button>
              Loop
            </label>
            <span className="text-[#8c8c8d]">Duration:</span>
            <input
              type="number"
              value={duration}
              onChange={e => updateDuration(parseFloat(e.target.value) || 1)}
              onPointerDown={e => e.stopPropagation()}
              className="w-[50px] bg-[#1a1a1a] border border-[rgba(255,255,255,0.06)] rounded px-1.5 py-0.5 text-[10px] text-[#fafafa] text-center appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/60 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            />
            <span className="text-[#666]">s</span>
          </div>

          {/* Input handles row */}
          <div ref={setRowRef("start")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>start</span>
          </div>
          <div ref={setRowRef("reset")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>reset</span>
          </div>

          {/* Trigger list */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-[#8c8c8d] font-medium">
              Triggers
            </span>
            <button
              className="px-2 py-0.5 rounded text-[9px] font-medium bg-[#333] text-[#999] hover:bg-[#444] hover:text-[#ccc] transition-colors"
              onClick={addTrigger}
              onPointerDown={e => e.stopPropagation()}
            >
              + Add
            </button>
          </div>

          {triggers.map((t, i) => (
            <div
              key={i}
              ref={
                dynamicPortNames.includes(t.port_name)
                  ? setRowRef(t.port_name)
                  : undefined
              }
              className="flex items-center gap-1.5 h-[22px]"
            >
              <NodePillInput
                type="text"
                value={t.port_name}
                onChange={v => updateTrigger(i, "port_name", String(v))}
              />
              <span className="text-[10px] text-[#666]">@</span>
              <input
                type="number"
                value={t.time}
                step={0.1}
                onChange={e =>
                  updateTrigger(i, "time", parseFloat(e.target.value) || 0)
                }
                onPointerDown={e => e.stopPropagation()}
                className="w-[48px] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5 text-[10px] text-[#fafafa] text-center appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/50 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
              <span className="text-[9px] text-[#666]">s</span>
              <button
                className="ml-auto text-[10px] text-[#666] hover:text-red-400 transition-colors"
                onClick={() => removeTrigger(i)}
                onPointerDown={e => e.stopPropagation()}
              >
                {"\u00D7"}
              </button>
            </div>
          ))}

          {/* Static output rows */}
          {staticOutputs.map(name => (
            <div
              key={name}
              ref={setRowRef(name)}
              className={NODE_TOKENS.paramRow}
            >
              <span className={NODE_TOKENS.labelText}>{name}</span>
              <span className="text-[10px] text-[#666]">
                {formatValue(
                  name === "elapsed"
                    ? elapsed
                    : name === "is_playing"
                      ? isPlaying
                      : name === "tick"
                        ? tickCount
                        : undefined
                )}
              </span>
            </div>
          ))}
        </NodeBody>
      )}

      {/* Input handles */}
      {inputNames.map(name => (
        <Handle
          key={`in-${name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[name] ?? 0,
                  left: 0,
                  backgroundColor: COLOR_TRIGGER,
                }
          }
        />
      ))}

      {/* Output handles */}
      {allOutputs.map(name => (
        <Handle
          key={`out-${name}`}
          type="source"
          position={Position.Right}
          id={buildHandleId("param", name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("right"), opacity: 0 }
              : {
                  top: rowPositions[name] ?? 0,
                  right: 0,
                  backgroundColor: outputColor(name),
                  boxShadow: flashingPorts.has(name)
                    ? "0 0 6px 2px rgba(255,255,255,0.6)"
                    : "none",
                  transition: "background-color 200ms, box-shadow 200ms",
                }
          }
        />
      ))}
    </NodeCard>
  );
}

function formatValue(v: unknown): string {
  if (v === undefined || v === null) return "\u2014";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}
