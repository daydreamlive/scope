import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId, parseHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
} from "../ui";

type ControlNodeType = Node<FlowNodeData, "control">;

const PARAM_TYPE_COLORS: Record<string, string> = {
  number: "#38bdf8", // sky-400 (for float and int)
  string: "#fbbf24", // amber-400
};

function getControlOutputType(
  controlType: "float" | "int" | "string"
): "number" | "string" {
  return controlType === "string" ? "string" : "number";
}

function getControlTypeColor(controlType: "float" | "int" | "string"): string {
  const outputType = getControlOutputType(controlType);
  return PARAM_TYPE_COLORS[outputType] || "#9ca3af";
}

function getControlTitle(type: "float" | "int" | "string"): string {
  if (type === "float") return "FloatControl";
  if (type === "int") return "IntControl";
  return "StringControl";
}

const PATTERN_OPTIONS = [
  { value: "sine", label: "Sine" },
  { value: "bounce", label: "Bounce" },
  { value: "random_walk", label: "Random Walk" },
  { value: "linear", label: "Linear" },
  { value: "step", label: "Step" },
];

const MODE_OPTIONS = [
  { value: "animated", label: "Animated" },
  { value: "switch", label: "Switch" },
];

function computePatternValue(
  pattern: "sine" | "bounce" | "random_walk" | "linear" | "step",
  t: number,
  speed: number,
  min: number,
  max: number,
  lastValue: number
): number {
  const range = max - min;
  const phase = (t * speed) % 1;

  switch (pattern) {
    case "sine":
      return min + range * (0.5 + 0.5 * Math.sin(phase * 2 * Math.PI));
    case "bounce": {
      const triangle = phase < 0.5 ? phase * 2 : 2 - phase * 2;
      return min + range * triangle;
    }
    case "random_walk": {
      const step = (Math.random() - 0.5) * 0.1 * range;
      const newValue = lastValue + step;
      return Math.max(min, Math.min(max, newValue));
    }
    case "linear":
      return min + range * phase;
    case "step": {
      const steps = 10;
      const stepIndex = Math.floor(phase * steps);
      return min + (range * stepIndex) / (steps - 1);
    }
    default:
      return min;
  }
}

function getNumberFromNode(
  node: Node<FlowNodeData>,
  sourceHandleId?: string | null
): number | null {
  if (node.data.nodeType === "primitive" || node.data.nodeType === "reroute") {
    const val = node.data.value;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "control" || node.data.nodeType === "math") {
    const val = node.data.currentValue;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "slider") {
    const val = node.data.value;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "knobs") {
    const knobs = node.data.knobs;
    if (!knobs || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("knob_", ""), 10);
    if (isNaN(idx) || idx >= knobs.length) return null;
    return knobs[idx].value;
  }
  if (node.data.nodeType === "xypad") {
    if (!sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    if (parsed.name === "x") return node.data.padX ?? null;
    if (parsed.name === "y") return node.data.padY ?? null;
    return null;
  }
  if (node.data.nodeType === "midi") {
    const channels = node.data.midiChannels;
    if (!channels || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("midi_", ""), 10);
    if (isNaN(idx) || idx >= channels.length) return null;
    return channels[idx].value;
  }
  if (node.data.nodeType === "bool") {
    const val = node.data.value;
    if (typeof val === "boolean") return val ? 1 : 0;
    return null;
  }
  return null;
}

function getStringFromNode(node: Node<FlowNodeData>): string | null {
  if (node.data.nodeType === "primitive" || node.data.nodeType === "reroute") {
    const val = node.data.value;
    if (typeof val === "string") return val;
    return null;
  }
  if (node.data.nodeType === "control") {
    const val = node.data.currentValue;
    if (typeof val === "string") return val;
    return null;
  }
  return null;
}

export function ControlNode({
  id,
  data,
  selected,
}: NodeProps<ControlNodeType>) {
  const { updateData: updateNodeData } = useNodeData(id);
  const controlType = data.controlType || "float";
  const pattern = data.controlPattern || "sine";
  const speed = data.controlSpeed ?? 1.0;
  const min = data.controlMin ?? 0;
  const max = data.controlMax ?? 1.0;
  const items = data.controlItems || ["item1", "item2", "item3"];
  const isPlaying = data.isPlaying ?? false;
  const controlMode = data.controlMode || "animated";
  const isSwitchMode = controlType === "string" && controlMode === "switch";

  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const [currentValue, setCurrentValue] = useState<number | string>(
    controlType === "string" ? items[0] || "" : min
  );
  const lastValueRef = useRef<number>(min);
  const startTimeRef = useRef<number>(Date.now());
  const animationFrameRef = useRef<number | undefined>(undefined);

  const color = getControlTypeColor(controlType);
  const dotColorClass = "bg-purple-400";
  const title = getControlTitle(controlType);

  // Initialize currentValue
  useEffect(() => {
    if (data.currentValue !== undefined) return;
    const initialValue = controlType === "string" ? items[0] || "" : min;
    updateNodeData({ currentValue: initialValue });
  }, [data.currentValue, updateNodeData, controlType, items, min]);

  // Switch mode: read string + number inputs per slot
  const switchSlots = isSwitchMode
    ? items.map((fallbackText, i) => {
        const strHandleId = buildHandleId("param", `str_${i}`);
        const strEdge = edges.find(
          e => e.target === id && e.targetHandle === strHandleId
        );
        const strSourceNode = strEdge
          ? allNodes.find(n => n.id === strEdge.source)
          : null;
        const connectedString = strSourceNode
          ? getStringFromNode(strSourceNode)
          : null;
        const text = connectedString !== null ? connectedString : fallbackText;
        const hasStringConnection = connectedString !== null;
        const numHandleId = buildHandleId("param", `item_${i}`);
        const numEdge = edges.find(
          e => e.target === id && e.targetHandle === numHandleId
        );
        const numSourceNode = numEdge
          ? allNodes.find(n => n.id === numEdge.source)
          : null;
        const numVal = numSourceNode
          ? (getNumberFromNode(numSourceNode, numEdge?.sourceHandle) ?? 0)
          : 0;

        return { text, numVal, hasStringConnection };
      })
    : [];

  const lastActiveIndexRef = useRef<number>(0);

  // Compute switch selection during render (produces a stable string primitive)
  let switchSelectedString: string | undefined;
  if (isSwitchMode && switchSlots.length > 0) {
    let bestIdx = lastActiveIndexRef.current;
    let bestVal = 0;
    for (let i = 0; i < switchSlots.length; i++) {
      if (switchSlots[i].numVal > bestVal) {
        bestVal = switchSlots[i].numVal;
        bestIdx = i;
      }
    }
    if (bestIdx >= switchSlots.length) bestIdx = 0;
    lastActiveIndexRef.current = bestIdx;
    switchSelectedString = switchSlots[bestIdx].text || "";
  }

  // Sync selection to node data — depends on a string, not the switchSlots array
  useEffect(() => {
    if (switchSelectedString === undefined) return;
    if (switchSelectedString !== data.currentValue) {
      updateNodeData({ currentValue: switchSelectedString });
    }
  }, [switchSelectedString, data.currentValue, updateNodeData]);

  // Animated mode
  useEffect(() => {
    if (isSwitchMode) return;
    if (!isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    const animate = () => {
      const now = Date.now();
      const elapsed = (now - startTimeRef.current) / 1000;

      if (controlType === "string") {
        const patternValue = computePatternValue(
          pattern,
          elapsed,
          speed,
          0,
          items.length - 1,
          lastValueRef.current
        );
        lastValueRef.current = patternValue;
        const index = Math.floor(patternValue);
        const clampedIndex = Math.max(0, Math.min(items.length - 1, index));
        setCurrentValue(items[clampedIndex] || "");
      } else {
        const floatValue = computePatternValue(
          pattern,
          elapsed,
          speed,
          min,
          max,
          lastValueRef.current
        );
        lastValueRef.current = floatValue;
        const finalValue =
          controlType === "int" ? Math.round(floatValue) : floatValue;
        setCurrentValue(finalValue);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isSwitchMode, isPlaying, pattern, speed, min, max, controlType, items]);

  const lastUpdateTimeRef = useRef<number>(0);
  useEffect(() => {
    if (isSwitchMode) return;
    const now = Date.now();
    if (now - lastUpdateTimeRef.current < 100) return; // Throttle to 100ms
    lastUpdateTimeRef.current = now;
    updateNodeData({ currentValue });
  }, [isSwitchMode, currentValue, updateNodeData]);

  const handleTogglePlay = () => {
    const newIsPlaying = !isPlaying;
    if (newIsPlaying) {
      startTimeRef.current = Date.now();
      lastValueRef.current =
        typeof currentValue === "number" ? currentValue : min;
    }
    updateNodeData({ isPlaying: newIsPlaying });
  };

  const handlePatternChange = (newPattern: string) => {
    updateNodeData({ controlPattern: newPattern as typeof pattern });
  };

  const handleMinChange = (val: string | number) => {
    updateNodeData({ controlMin: Number(val) });
  };

  const handleMaxChange = (val: string | number) => {
    updateNodeData({ controlMax: Number(val) });
  };

  const handleSpeedChange = (val: string | number) => {
    updateNodeData({ controlSpeed: Number(val) });
  };

  const handleItemsChange = (val: string | number) => {
    const itemsStr = String(val);
    const itemsArray = itemsStr
      .split(",")
      .map(s => s.trim())
      .filter(s => s.length > 0);
    updateNodeData({
      controlItems: itemsArray.length > 0 ? itemsArray : ["item1"],
    });
  };

  const handleModeChange = (newMode: string) => {
    updateNodeData({
      controlMode: newMode as "animated" | "switch",
      isPlaying: false,
    });
  };

  // Add / remove item slots in switch mode
  const handleAddItem = () => {
    const newItems = [...items, `item${items.length + 1}`];
    updateNodeData({ controlItems: newItems });
  };

  const handleRemoveItem = (index: number) => {
    if (items.length <= 1) return;
    const newItems = items.filter((_, i) => i !== index);
    updateNodeData({ controlItems: newItems });
  };

  // Edit individual item text (when not string-connected)
  const handleItemTextChange = (index: number, text: string) => {
    const newItems = [...items];
    newItems[index] = text;
    updateNodeData({ controlItems: newItems });
  };

  const itemsDisplay = items.join(", ");

  // Measure handle positions for switch mode
  const { setRowRef, rowPositions } = useHandlePositions([
    isSwitchMode,
    items.length,
  ]);

  return (
    <NodeCard selected={selected} autoMinHeight={isSwitchMode}>
      <NodeHeader
        title={data.customTitle || title}
        dotColor={dotColorClass}
        onTitleChange={newTitle => updateNodeData({ customTitle: newTitle })}
        rightContent={
          !isSwitchMode ? (
            <button
              onClick={handleTogglePlay}
              className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors"
              type="button"
            >
              {isPlaying ? "⏸" : "▶"}
            </button>
          ) : undefined
        }
      />
      <NodeBody withGap>
        {/* Mode selector — only for string type */}
        {controlType === "string" && (
          <NodeParamRow label="Mode">
            <NodePillSelect
              value={controlMode}
              onChange={handleModeChange}
              options={MODE_OPTIONS}
            />
          </NodeParamRow>
        )}

        {/* Animated mode controls */}
        {!isSwitchMode && (
          <>
            <NodeParamRow label="Pattern">
              <NodePillSelect
                value={pattern}
                onChange={handlePatternChange}
                options={PATTERN_OPTIONS}
              />
            </NodeParamRow>

            {controlType === "string" ? (
              <NodeParamRow label="Items">
                <NodePillInput
                  type="text"
                  value={itemsDisplay}
                  onChange={handleItemsChange}
                />
              </NodeParamRow>
            ) : (
              <>
                <NodeParamRow label="Min">
                  <NodePillInput
                    type="number"
                    value={min}
                    onChange={handleMinChange}
                  />
                </NodeParamRow>
                <NodeParamRow label="Max">
                  <NodePillInput
                    type="number"
                    value={max}
                    onChange={handleMaxChange}
                  />
                </NodeParamRow>
              </>
            )}

            <NodeParamRow label="Speed">
              <NodePillInput
                type="number"
                value={speed}
                onChange={handleSpeedChange}
                min={0.1}
              />
            </NodeParamRow>
          </>
        )}

        {/* Switch mode: per-item rows with string + number handles */}
        {isSwitchMode && (
          <>
            {switchSlots.map((slot, i) => {
              const isSelected = data.currentValue === slot.text;
              const isActive = isSelected && slot.numVal > 0;
              return (
                <div
                  key={i}
                  ref={setRowRef(`item_${i}`)}
                  className="flex items-center gap-1 min-h-[22px]"
                >
                  {/* Activity dot */}
                  <div
                    className="w-2 h-2 rounded-full shrink-0 transition-colors"
                    style={{
                      backgroundColor: isActive
                        ? "#fbbf24"
                        : isSelected
                          ? "#fbbf24"
                          : "#333",
                      opacity: isActive ? 1 : isSelected ? 0.5 : 1,
                      boxShadow: isActive ? "0 0 6px #fbbf24" : "none",
                    }}
                  />

                  {/* Text: editable input if no string connection, else display connected text */}
                  {slot.hasStringConnection ? (
                    <span
                      className={`text-[10px] font-medium truncate flex-1 min-w-0 ${
                        isSelected ? "text-amber-400" : "text-[#8c8c8d]"
                      }`}
                      title={slot.text}
                    >
                      {slot.text || "—"}
                    </span>
                  ) : (
                    <input
                      className={`${NODE_TOKENS.pillInput} !w-auto flex-1 min-w-0 !text-[9px] !px-1.5 !py-0 ${
                        isSelected ? "!text-amber-400" : ""
                      }`}
                      value={items[i]}
                      onChange={e => handleItemTextChange(i, e.target.value)}
                      onMouseDown={e => e.stopPropagation()}
                      title={items[i]}
                    />
                  )}

                  {/* MIDI value indicator */}
                  <span className="text-[9px] text-[#666] w-[30px] text-right shrink-0">
                    {slot.numVal > 0 ? slot.numVal.toFixed(1) : ""}
                  </span>

                  {/* Remove button */}
                  {items.length > 1 && (
                    <button
                      className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0"
                      onClick={() => handleRemoveItem(i)}
                    >
                      ×
                    </button>
                  )}
                </div>
              );
            })}

            {/* Add button */}
            <button
              className="w-full py-0.5 mt-0.5 rounded bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-[#ccc] hover:border-[rgba(119,119,119,0.4)] text-[10px] transition-colors"
              onClick={handleAddItem}
            >
              + Add Item
            </button>
          </>
        )}

        {/* Current value display */}
        <div ref={setRowRef("value")} className={NODE_TOKENS.paramRow}>
          <span className={NODE_TOKENS.labelText}>Value</span>
          <NodePill className="opacity-75">
            {(() => {
              const displayVal = isSwitchMode
                ? (switchSelectedString ?? String(data.currentValue ?? ""))
                : currentValue;
              if (typeof displayVal === "number") {
                return controlType === "int"
                  ? Math.round(displayVal)
                  : displayVal.toFixed(3);
              }
              const s = String(displayVal);
              return s.length > 20 ? s.slice(0, 20) + "…" : s;
            })()}
          </NodePill>
        </div>
      </NodeBody>

      {isSwitchMode &&
        items.map((_, i) => (
          <span key={`handles_${i}`}>
            <Handle
              type="target"
              position={Position.Left}
              id={buildHandleId("param", `str_${i}`)}
              className="!w-2 !h-2 !border-0"
              style={{
                top: rowPositions[`item_${i}`] ?? 78 + i * 24,
                left: -1,
                backgroundColor: "#fbbf24",
              }}
            />
            <Handle
              type="target"
              position={Position.Left}
              id={buildHandleId("param", `item_${i}`)}
              className="!w-2 !h-2 !border-0"
              style={{
                top: rowPositions[`item_${i}`] ?? 78 + i * 24,
                left: 10,
                backgroundColor: "#38bdf8",
              }}
            />
          </span>
        ))}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["value"] ?? 44,
          right: 8,
          backgroundColor: color,
        }}
      />
    </NodeCard>
  );
}
