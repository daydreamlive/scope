import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId, parseHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
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

function isUnaryOp(op: string): boolean {
  return UNARY_OPS.has(op);
}

function getValueFromNode(
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
  return null;
}

function computeResult(
  op: string,
  a: number | null,
  b: number | null
): number | null {
  if (a === null) return null;

  // Unary ops
  switch (op) {
    case "abs":
      return Math.abs(a);
    case "negate":
      return -a;
    case "sqrt":
      return a >= 0 ? Math.sqrt(a) : null;
    case "floor":
      return Math.floor(a);
    case "ceil":
      return Math.ceil(a);
    case "round":
      return Math.round(a);
    case "toInt":
      return Math.trunc(a);
    case "toFloat":
      return a + 0.0; // identity — ensures float representation
  }

  // Binary ops
  if (b === null) return null;

  switch (op) {
    case "add":
      return a + b;
    case "subtract":
      return a - b;
    case "multiply":
      return a * b;
    case "divide":
      return b !== 0 ? a / b : null;
    case "mod":
      return b !== 0 ? a % b : null;
    case "min":
      return Math.min(a, b);
    case "max":
      return Math.max(a, b);
    case "power":
      return Math.pow(a, b);
    default:
      return null;
  }
}

export function MathNode({ id, data, selected }: NodeProps<MathNodeType>) {
  const { updateData } = useNodeData(id);
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];
  const operation = data.mathOp || "add";
  const unary = isUnaryOp(operation);

  // Find input edges
  const edgeA = edges.find(
    e => e.target === id && e.targetHandle === buildHandleId("param", "a")
  );
  const edgeB = !unary
    ? edges.find(
        e => e.target === id && e.targetHandle === buildHandleId("param", "b")
      )
    : null;

  // Get source nodes
  const sourceNodeA = edgeA ? allNodes.find(n => n.id === edgeA.source) : null;
  const sourceNodeB = edgeB ? allNodes.find(n => n.id === edgeB.source) : null;

  // Extract values
  const valueA = sourceNodeA
    ? getValueFromNode(sourceNodeA, edgeA?.sourceHandle)
    : null;
  const valueB = sourceNodeB
    ? getValueFromNode(sourceNodeB, edgeB?.sourceHandle)
    : null;

  // Compute
  const result = computeResult(operation, valueA, valueB);

  // Update currentValue
  useEffect(() => {
    updateData({ currentValue: result ?? undefined });
  }, [updateData, result]);

  const handleOperationChange = (newOp: string) => {
    updateData({ mathOp: newOp as typeof operation });
  };

  // Measure handle positions when operation type changes
  const { setRowRef, rowPositions } = useHandlePositions([unary]);

  return (
    <NodeCard selected={selected}>
      <NodeHeader
        title={data.customTitle || "Math"}
        dotColor="bg-sky-400"
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      <NodeBody withGap>
        <NodeParamRow label="Op">
          <NodePillSelect
            value={operation}
            onChange={handleOperationChange}
            options={ALL_OPERATIONS}
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
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "a")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["a"] ?? 56,
          left: 8,
          backgroundColor: COLOR,
        }}
      />
      {!unary && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "b")}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions["b"] ?? 78,
            left: 8,
            backgroundColor: COLOR,
          }}
        />
      )}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["result"] ?? (unary ? 78 : 100),
          right: 8,
          backgroundColor: COLOR,
        }}
      />
    </NodeCard>
  );
}
