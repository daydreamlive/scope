import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import {
  getNumberFromNode,
  getStringFromNode,
} from "../utils/getValueFromNode";
import { computePatternValue } from "../utils/computePatternValue";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { useConnectedNumber } from "../hooks/node/useConnectedValue";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
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

export function ControlNode({
  id,
  data,
  selected,
}: NodeProps<ControlNodeType>) {
  const { updateData: updateNodeData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
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

  const speedIn = useConnectedNumber(id, "speed", speed);
  const effectiveSpeed = speedIn.value;
  const minIn = useConnectedNumber(id, "min", min);
  const effectiveMin = minIn.value;
  const maxIn = useConnectedNumber(id, "max", max);
  const effectiveMax = maxIn.value;

  const [currentValue, setCurrentValue] = useState<number | string>(
    controlType === "string" ? items[0] || "" : min
  );
  const lastValueRef = useRef<number>(min);
  const startTimeRef = useRef<number>(Date.now());
  const animationFrameRef = useRef<number | undefined>(undefined);

  const color = getControlTypeColor(controlType);
  const title = getControlTitle(controlType);

  useEffect(() => {
    if (data.currentValue !== undefined) return;
    const initialValue = controlType === "string" ? items[0] || "" : min;
    updateNodeData({ currentValue: initialValue });
  }, [data.currentValue, updateNodeData, controlType, items, min]);

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
          ? getStringFromNode(strSourceNode, strEdge?.sourceHandle)
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

  useEffect(() => {
    if (switchSelectedString === undefined) return;
    if (switchSelectedString !== data.currentValue) {
      updateNodeData({ currentValue: switchSelectedString });
    }
  }, [switchSelectedString, data.currentValue, updateNodeData]);

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
          effectiveSpeed,
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
          effectiveSpeed,
          effectiveMin,
          effectiveMax,
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
  }, [
    isSwitchMode,
    isPlaying,
    pattern,
    effectiveSpeed,
    effectiveMin,
    effectiveMax,
    controlType,
    items,
  ]);

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

  const handleItemTextChange = (index: number, text: string) => {
    const newItems = [...items];
    newItems[index] = text;
    updateNodeData({ controlItems: newItems });
  };

  const itemsDisplay = items.join(", ");

  const { setRowRef, rowPositions } = useHandlePositions([
    isSwitchMode,
    items.length,
    speedIn.connected,
    minIn.connected,
    maxIn.connected,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={isSwitchMode && !collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || title}
        onTitleChange={newTitle => updateNodeData({ customTitle: newTitle })}
        rightContent={
          !isSwitchMode && !collapsed ? (
            <button
              onClick={handleTogglePlay}
              className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors"
              type="button"
            >
              {isPlaying ? "⏸" : "▶"}
            </button>
          ) : undefined
        }
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
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
                  <div ref={setRowRef("min")}>
                    <NodeParamRow label="Min">
                      {minIn.connected ? (
                        <NodePill className="opacity-50">
                          {effectiveMin}
                        </NodePill>
                      ) : (
                        <NodePillInput
                          type="number"
                          value={min}
                          onChange={handleMinChange}
                        />
                      )}
                    </NodeParamRow>
                  </div>
                  <div ref={setRowRef("max")}>
                    <NodeParamRow label="Max">
                      {maxIn.connected ? (
                        <NodePill className="opacity-50">
                          {effectiveMax}
                        </NodePill>
                      ) : (
                        <NodePillInput
                          type="number"
                          value={max}
                          onChange={handleMaxChange}
                        />
                      )}
                    </NodeParamRow>
                  </div>
                </>
              )}

              <div ref={setRowRef("speed")}>
                <NodeParamRow label="Speed">
                  {speedIn.connected ? (
                    <NodePill className="opacity-50">
                      {effectiveSpeed.toFixed(2)}
                    </NodePill>
                  ) : (
                    <NodePillInput
                      type="number"
                      value={speed}
                      onChange={handleSpeedChange}
                      min={0.1}
                    />
                  )}
                </NodeParamRow>
              </div>
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
      )}

      {isSwitchMode &&
        items.map((_, i) => (
          <span key={`handles_${i}`}>
            <Handle
              type="target"
              position={Position.Left}
              id={buildHandleId("param", `str_${i}`)}
              className={
                collapsed && i > 0
                  ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                  : "!w-2.5 !h-2.5 !border-0"
              }
              style={
                collapsed
                  ? i === 0
                    ? collapsedHandleStyle("left")
                    : { ...collapsedHandleStyle("left"), opacity: 0 }
                  : {
                      top: (rowPositions[`item_${i}`] ?? 78 + i * 24) - 5,
                      left: 0,
                      backgroundColor: "#fbbf24",
                    }
              }
            />
            <Handle
              type="target"
              position={Position.Left}
              id={buildHandleId("param", `item_${i}`)}
              className={
                collapsed
                  ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                  : "!w-2.5 !h-2.5 !border-0"
              }
              style={
                collapsed
                  ? { ...collapsedHandleStyle("left"), opacity: 0 }
                  : {
                      top: (rowPositions[`item_${i}`] ?? 78 + i * 24) + 5,
                      left: 0,
                      backgroundColor: "#38bdf8",
                    }
              }
            />
          </span>
        ))}

      {/* Param input handles (animated mode only) */}
      {!isSwitchMode && (
        <>
          <Handle
            type="target"
            position={Position.Left}
            id={buildHandleId("param", "min")}
            className={
              collapsed
                ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                : "!w-2.5 !h-2.5 !border-0"
            }
            style={
              collapsed
                ? { ...collapsedHandleStyle("left"), opacity: 0 }
                : {
                    top: rowPositions["min"] ?? 44,
                    left: 0,
                    backgroundColor: "#38bdf8",
                  }
            }
          />
          <Handle
            type="target"
            position={Position.Left}
            id={buildHandleId("param", "max")}
            className={
              collapsed
                ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                : "!w-2.5 !h-2.5 !border-0"
            }
            style={
              collapsed
                ? { ...collapsedHandleStyle("left"), opacity: 0 }
                : {
                    top: rowPositions["max"] ?? 44,
                    left: 0,
                    backgroundColor: "#38bdf8",
                  }
            }
          />
          <Handle
            type="target"
            position={Position.Left}
            id={buildHandleId("param", "speed")}
            className={
              collapsed
                ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                : "!w-2.5 !h-2.5 !border-0"
            }
            style={
              collapsed
                ? { ...collapsedHandleStyle("left"), opacity: 0 }
                : {
                    top: rowPositions["speed"] ?? 44,
                    left: 0,
                    backgroundColor: "#38bdf8",
                  }
            }
          />
        </>
      )}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["value"] ?? 44,
                right: 0,
                backgroundColor: color,
              }
        }
      />
    </NodeCard>
  );
}
