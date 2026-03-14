import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getNumberFromNode } from "../utils/getValueFromNode";
import { computeResult } from "../utils/computeResult";
import { useNodeData } from "../hooks/useNodeData";
import { useNodeCollapse } from "../hooks/useNodeCollapse";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type MathNodeType = Node<FlowNodeData, "math">;

const BINARY_OPERATIONS = [
  { value: "add", label: "Add" },
  { value: "subtract", label: "Subtract" },
  { value: "multiply", label: "Multiply" },
  { value: "divide", label: "Divide" },
  { value: "mod", label: "Mod" },
  { value: "min", label: "Min" },
  { value: "max", label: "Max" },
  { value: "power", label: "Power" },
];

const UNARY_OPERATIONS = [
  { value: "abs", label: "Abs" },
  { value: "negate", label: "Negate" },
  { value: "sqrt", label: "Sqrt" },
  { value: "floor", label: "Floor" },
  { value: "ceil", label: "Ceil" },
  { value: "round", label: "Round" },
  { value: "toInt", label: "Float → Int" },
  { value: "toFloat", label: "Int → Float" },
];

const ALL_OPERATIONS = [...BINARY_OPERATIONS, ...UNARY_OPERATIONS];

const UNARY_OPS = new Set(UNARY_OPERATIONS.map(o => o.value));
const COLOR = "#38bdf8"; // sky-400

const OUTPUT_TYPE_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "int", label: "Int" },
  { value: "float", label: "Float" },
];

function isUnaryOp(op: string): boolean {
  return UNARY_OPS.has(op);
}

export function MathNode({ id, data, selected }: NodeProps<MathNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];
  const operation = data.mathOp || "add";
  const unary = isUnaryOp(operation);

  const edgeA = edges.find(
    e => e.target === id && e.targetHandle === buildHandleId("param", "a")
  );
  const edgeB = !unary
    ? edges.find(
        e => e.target === id && e.targetHandle === buildHandleId("param", "b")
      )
    : null;

  const sourceNodeA = edgeA ? allNodes.find(n => n.id === edgeA.source) : null;
  const sourceNodeB = edgeB ? allNodes.find(n => n.id === edgeB.source) : null;

  const valueA = sourceNodeA
    ? getNumberFromNode(sourceNodeA, edgeA?.sourceHandle)
    : null;
  const valueB = sourceNodeB
    ? getNumberFromNode(sourceNodeB, edgeB?.sourceHandle)
    : null;

  let result = computeResult(operation, valueA, valueB);

  const outputType = data.mathOutputType;
  if (result !== null && outputType) {
    if (outputType === "int") {
      result = Math.trunc(result);
    } else if (outputType === "float") {
      result = result + 0.0; // Ensure float representation
    }
  }

  const rafRef = useRef<number>(0);
  useEffect(() => {
    const next = result ?? undefined;
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      updateData({ currentValue: next });
    });
    return () => cancelAnimationFrame(rafRef.current);
  }, [updateData, result]);

  const handleOperationChange = (newOp: string) => {
    updateData({ mathOp: newOp as typeof operation });
  };

  const handleOutputTypeChange = (newType: string) => {
    updateData({
      mathOutputType: newType === "auto" ? null : (newType as "int" | "float"),
    });
  };

  const { setRowRef, rowPositions } = useHandlePositions([unary]);

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Math"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          <NodeParamRow label="Op">
            <NodePillSelect
              value={operation}
              onChange={handleOperationChange}
              options={ALL_OPERATIONS}
            />
          </NodeParamRow>
          <NodeParamRow label="Output">
            <NodePillSelect
              value={outputType || "auto"}
              onChange={handleOutputTypeChange}
              options={OUTPUT_TYPE_OPTIONS}
            />
          </NodeParamRow>
          <div ref={setRowRef("a")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>A</span>
            <NodePill className="opacity-75">
              {valueA !== null ? valueA.toFixed(3) : "—"}
            </NodePill>
          </div>
          {!unary && (
            <div ref={setRowRef("b")} className={NODE_TOKENS.paramRow}>
              <span className={NODE_TOKENS.labelText}>B</span>
              <NodePill className="opacity-75">
                {valueB !== null ? valueB.toFixed(3) : "—"}
              </NodePill>
            </div>
          )}
          <div ref={setRowRef("result")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>Result</span>
            <NodePill className="opacity-75">
              {result !== null
                ? typeof result === "number" && !Number.isNaN(result)
                  ? result.toFixed(3)
                  : "Error"
                : "—"}
            </NodePill>
          </div>
        </NodeBody>
      )}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "a")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : {
                top: rowPositions["a"] ?? 56,
                left: 0,
                backgroundColor: COLOR,
              }
        }
      />
      {!unary && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "b")}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions["b"] ?? 78,
                  left: 0,
                  backgroundColor: COLOR,
                }
          }
        />
      )}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["result"] ?? (unary ? 78 : 100),
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}
