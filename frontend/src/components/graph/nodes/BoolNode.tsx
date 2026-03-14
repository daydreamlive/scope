import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getNumberFromNode } from "../utils/getValueFromNode";
import { useNodeData } from "../hooks/useNodeData";
import { useNodeCollapse } from "../hooks/useNodeCollapse";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePillInput,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type BoolNodeType = Node<FlowNodeData, "bool">;

const COLOR = "#34d399"; // emerald-400

const MODE_OPTIONS = [
  { value: "gate", label: "Gate" },
  { value: "toggle", label: "Toggle" },
];

export function BoolNode({ id, data, selected }: NodeProps<BoolNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const mode = data.boolMode || "gate";
  const threshold = data.boolThreshold ?? 0;
  const currentOutput = Boolean(data.value);

  const inputEdge = edges.find(
    e => e.target === id && e.targetHandle === buildHandleId("param", "input")
  );

  const sourceNode = inputEdge
    ? allNodes.find(n => n.id === inputEdge.source)
    : null;
  const inputValue = sourceNode
    ? getNumberFromNode(sourceNode, inputEdge?.sourceHandle)
    : null;

  const isAboveThreshold = inputValue !== null && inputValue > threshold;

  // rising-edge ref for toggle mode
  const prevAboveRef = useRef(false);
  const toggleStateRef = useRef(currentOutput);

  useEffect(() => {
    let newOutput: boolean;

    if (mode === "gate") {
      newOutput = isAboveThreshold;
    } else {
      // Toggle: flip on rising edge
      const wasAbove = prevAboveRef.current;
      if (isAboveThreshold && !wasAbove) {
        toggleStateRef.current = !toggleStateRef.current;
      }
      prevAboveRef.current = isAboveThreshold;
      newOutput = toggleStateRef.current;
    }

    if (newOutput !== currentOutput) {
      updateData({ value: newOutput });
    }
  }, [mode, isAboveThreshold, currentOutput, updateData]);

  // Sync toggle ref if data.value changes externally
  useEffect(() => {
    toggleStateRef.current = Boolean(data.value);
  }, [data.value]);

  // Sync prevAboveRef in gate mode
  useEffect(() => {
    if (mode === "gate") {
      prevAboveRef.current = isAboveThreshold;
    }
  }, [mode, isAboveThreshold]);

  const { setRowRef, rowPositions } = useHandlePositions([mode]);

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Bool"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          <NodeParamRow label="Mode">
            <NodePillSelect
              value={mode}
              onChange={v => updateData({ boolMode: v as "gate" | "toggle" })}
              options={MODE_OPTIONS}
            />
          </NodeParamRow>
          <NodeParamRow label="Threshold">
            <NodePillInput
              type="number"
              value={threshold}
              onChange={v => updateData({ boolThreshold: Number(v) })}
            />
          </NodeParamRow>
          <div ref={setRowRef("input")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>In</span>
            <NodePill className="opacity-75">
              {inputValue !== null ? inputValue.toFixed(3) : "—"}
            </NodePill>
          </div>
          <div ref={setRowRef("output")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>Out</span>
            <div className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full transition-colors"
                style={{
                  backgroundColor: currentOutput ? COLOR : "#333",
                  boxShadow: currentOutput ? `0 0 6px ${COLOR}` : "none",
                }}
              />
              <span
                className={`text-[10px] font-medium ${currentOutput ? "text-emerald-400" : "text-[#666]"}`}
              >
                {currentOutput ? "true" : "false"}
              </span>
            </div>
          </div>
        </NodeBody>
      )}

      {/* Input handle (number) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "input")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : {
                top: rowPositions["input"] ?? 78,
                left: 0,
                backgroundColor: "#38bdf8", // sky-400, number color
              }
        }
      />

      {/* Output handle (boolean) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["output"] ?? 100,
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}
