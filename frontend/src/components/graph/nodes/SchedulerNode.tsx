import { Handle, Position } from "@xyflow/react";
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

type SchedulerNodeType = Node<FlowNodeData, "scheduler_node">;

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

  const nodeState = (data.backendNodeState ?? {}) as Record<string, unknown>;
  const isStreaming = (data.isStreaming as boolean) ?? false;
  const sendInput = data.onBackendNodeInput as
    | ((name: string, value: unknown) => void)
    | undefined;
  const sendConfig = data.onBackendNodeConfig as
    | ((config: Record<string, unknown>) => void)
    | undefined;
  const sendConfigRef = useRef(sendConfig);
  sendConfigRef.current = sendConfig;
  const hasSendConfig = !!sendConfig;

  const isPlaying = (nodeState.is_playing as boolean) ?? false;
  const elapsed = (nodeState.elapsed as number) ?? 0;

  const rawBackendTriggers = nodeState.triggers as TriggerEntry[] | undefined;
  const backendTriggers = useMemo(
    () => rawBackendTriggers ?? [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rawBackendTriggers ? triggersKey(rawBackendTriggers) : ""]
  );
  const backendDuration = nodeState.duration as number | undefined;
  const backendLoop = nodeState.loop as boolean | undefined;
  const backendHasConfig = backendDuration !== undefined;

  const rawSavedTriggers = data.schedulerTriggers as TriggerEntry[] | undefined;
  const savedTriggers = useMemo(
    () => rawSavedTriggers ?? [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rawSavedTriggers ? triggersKey(rawSavedTriggers) : ""]
  );
  const savedDuration = (data.schedulerDuration as number) ?? 30;
  const savedLoop = (data.schedulerLoop as boolean) ?? false;

  const [triggers, setTriggers] = useState<TriggerEntry[]>(savedTriggers);
  const [duration, setDuration] = useState(savedDuration);
  const [loop, setLoop] = useState(savedLoop);
  const initializedRef = useRef(false);
  const lastPushedConfigRef = useRef("");

  useEffect(() => {
    if (initializedRef.current) {
      if (!backendHasConfig && savedTriggers.length > 0) {
        initializedRef.current = false;
        lastPushedConfigRef.current = "";
      } else {
        return;
      }
    }

    if (!isStreaming) return;

    if (backendHasConfig) {
      initializedRef.current = true;
      setTriggers(backendTriggers);
      if (typeof backendDuration === "number" && backendDuration > 0)
        setDuration(backendDuration);
      if (typeof backendLoop === "boolean") setLoop(backendLoop);
    } else if (savedTriggers.length > 0) {
      const cfg = sendConfigRef.current;
      if (!cfg) return;
      const configKey = `${triggersKey(savedTriggers)}:${savedDuration}:${savedLoop}`;
      if (lastPushedConfigRef.current === configKey) return;
      lastPushedConfigRef.current = configKey;
      initializedRef.current = true;
      setTriggers(savedTriggers);
      setDuration(savedDuration);
      setLoop(savedLoop);
      cfg({
        triggers: savedTriggers,
        duration: savedDuration,
        loop: savedLoop,
      });
    }
  }, [
    backendHasConfig,
    backendTriggers,
    backendDuration,
    backendLoop,
    savedTriggers,
    savedDuration,
    savedLoop,
    hasSendConfig,
    isStreaming,
  ]);

  const configTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pushConfig = useCallback(
    (t: TriggerEntry[], d: number, l: boolean) => {
      updateData({
        schedulerTriggers: t,
        schedulerDuration: d,
        schedulerLoop: l,
      });
      if (configTimerRef.current) clearTimeout(configTimerRef.current);
      configTimerRef.current = setTimeout(() => {
        sendConfig?.({ triggers: t, duration: d, loop: l });
      }, 300);
    },
    [sendConfig, updateData]
  );

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

  const addTrigger = useCallback(() => {
    const name = `trigger_${triggers.length + 1}`;
    const time = Math.min(duration, elapsed + 1);
    const next = [...triggers, { time, port_name: name }];
    setTriggers(next);
    pushConfig(next, duration, loop);
  }, [triggers, duration, elapsed, loop, pushConfig]);

  const removeTrigger = useCallback(
    (idx: number) => {
      const next = triggers.filter((_, i) => i !== idx);
      setTriggers(next);
      pushConfig(next, duration, loop);
    },
    [triggers, duration, loop, pushConfig]
  );

  const updateTrigger = useCallback(
    (idx: number, field: "time" | "port_name", value: string | number) => {
      const next = triggers.map((t, i) =>
        i === idx ? { ...t, [field]: value } : t
      );
      setTriggers(next);
      pushConfig(next, duration, loop);
    },
    [triggers, duration, loop, pushConfig]
  );

  const updateDuration = useCallback(
    (v: number) => {
      const d = Math.max(1, v);
      setDuration(d);
      pushConfig(triggers, d, loop);
    },
    [triggers, loop, pushConfig]
  );

  const toggleLoop = useCallback(() => {
    const l = !loop;
    setLoop(l);
    pushConfig(triggers, duration, l);
  }, [triggers, duration, loop, pushConfig]);

  const nodeName = data.label ?? "Scheduler";

  const [flashingPorts, setFlashingPorts] = useState<Set<string>>(new Set());
  const flashTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map()
  );
  const lastCounters = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const triggerPorts = ["tick", ...dynamicPortNames];
    for (const port of triggerPorts) {
      const counter = Number(nodeState[port]) || 0;
      const prev = lastCounters.current.get(port) ?? 0;
      if (counter > 0 && counter !== prev) {
        lastCounters.current.set(port, counter);
        setFlashingPorts(p => {
          if (p.has(port)) return p;
          const next = new Set(p);
          next.add(port);
          return next;
        });
        const existing = flashTimers.current.get(port);
        if (existing) clearTimeout(existing);
        flashTimers.current.set(
          port,
          setTimeout(() => {
            setFlashingPorts(p => {
              const next = new Set(p);
              next.delete(port);
              return next;
            });
            flashTimers.current.delete(port);
          }, 200)
        );
      }
    }
  }, [nodeState, dynamicPortNames]);

  function outputColor(name: string): string {
    if (flashingPorts.has(name)) return "#ffffff";
    if (name === "elapsed") return COLOR_FLOAT;
    if (name === "is_playing") return COLOR_BOOLEAN;
    return COLOR_TRIGGER;
  }

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
                  !isStreaming
                    ? "bg-[#333] text-[#555] cursor-not-allowed"
                    : isPlaying
                      ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                      : "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                }`}
                disabled={!isStreaming}
                onClick={() => sendInput?.("start", true)}
              >
                {isPlaying ? "■" : "▶"}
              </button>
              <button
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  !isStreaming
                    ? "bg-[#333] text-[#555] cursor-not-allowed"
                    : "bg-[#333] text-[#999] hover:bg-[#444] hover:text-[#ccc]"
                }`}
                disabled={!isStreaming}
                onClick={() => sendInput?.("reset", true)}
              >
                ↺
              </button>
            </div>
          )
        }
      />

      {!collapsed && (
        <NodeBody withGap>
          {/* Transport row */}
          <div className="flex items-center gap-2 text-[10px]">
            <label className="flex items-center gap-1 text-[#8c8c8d]">
              <input
                type="checkbox"
                checked={loop}
                onChange={toggleLoop}
                className="w-3 h-3 accent-blue-500"
              />
              Loop
            </label>
            <span className="text-[#8c8c8d]">Duration:</span>
            <input
              type="number"
              value={duration}
              onChange={e => updateDuration(parseFloat(e.target.value) || 1)}
              className="w-[50px] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded px-1.5 py-0.5 text-[10px] text-[#fafafa] text-center appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/50 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
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
                className="w-[48px] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5 text-[10px] text-[#fafafa] text-center appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/50 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
              <span className="text-[9px] text-[#666]">s</span>
              <button
                className="ml-auto text-[10px] text-[#666] hover:text-red-400 transition-colors"
                onClick={() => removeTrigger(i)}
              >
                ×
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
                {formatValue(nodeState[name])}
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
  if (v === undefined || v === null) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}
