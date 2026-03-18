import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { parseHandleId } from "../../lib/graphUtils";

export const HANDLE_COLORS: Record<string, string> = {
  video: "#eeeeee",
  video2: "#eeeeee",
  vace_input_frames: "#ffffff",
  vace_input_masks: "#f472b6",
  source: "#4ade80",
  sink: "#fb923c",
  record: "#ef4444",
};

export const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
  float: "#38bdf8",
  int: "#38bdf8",
  video_path: "#eeeeee",
};

export function getEdgeColor(
  sourceNode: Node<FlowNodeData> | undefined,
  handleId: string | null | undefined
): string {
  if (!sourceNode || !handleId) return "#9ca3af";

  const parsed = parseHandleId(handleId);
  if (!parsed) return "#9ca3af";

  if (parsed.kind === "param") {
    if (sourceNode.data.nodeType === "primitive") {
      const valueType = sourceNode.data.valueType;
      return PARAM_TYPE_COLORS[valueType || "string"] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "reroute") {
      const valueType = sourceNode.data.valueType;
      return valueType ? PARAM_TYPE_COLORS[valueType] || "#9ca3af" : "#9ca3af";
    }
    if (sourceNode.data.nodeType === "control") {
      const controlType = sourceNode.data.controlType;
      const outputType = controlType === "string" ? "string" : "number";
      return PARAM_TYPE_COLORS[outputType] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "math") {
      return PARAM_TYPE_COLORS["number"] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "slider") {
      return "#38bdf8"; // sky-400 (number)
    }
    if (sourceNode.data.nodeType === "knobs") {
      return "#38bdf8"; // sky-400 (number)
    }
    if (sourceNode.data.nodeType === "xypad") {
      return "#38bdf8"; // sky-400 (number)
    }
    if (sourceNode.data.nodeType === "tuple") {
      return "#fb923c"; // orange-400 (list_number)
    }
    if (sourceNode.data.nodeType === "image") {
      return sourceNode.data.mediaType === "video"
        ? "#eeeeee" // white for video_path
        : "#fbbf24"; // amber-400 for string
    }
    if (sourceNode.data.nodeType === "vace") {
      return "#a78bfa"; // violet-400 (vace compound)
    }
    if (sourceNode.data.nodeType === "midi") {
      return "#38bdf8"; // sky-400 (number)
    }
    if (sourceNode.data.nodeType === "bool") {
      return "#34d399"; // emerald-400
    }
    if (sourceNode.data.nodeType === "subgraph") {
      const port = sourceNode.data.subgraphOutputs?.find(
        p => p.name === parsed.name
      );
      if (port?.paramType) {
        return PARAM_TYPE_COLORS[port.paramType] || "#9ca3af";
      }
      return "#9ca3af";
    }
    if (sourceNode.data.nodeType === "subgraph_input") {
      const port = sourceNode.data.subgraphInputs?.find(
        p => p.name === parsed.name
      );
      if (port?.paramType) {
        return PARAM_TYPE_COLORS[port.paramType] || "#9ca3af";
      }
      return "#9ca3af";
    }
    return "#9ca3af";
  }

  if (sourceNode.data.nodeType === "pipeline") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "source") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }
  if (sourceNode.data.nodeType === "sink") {
    return HANDLE_COLORS.sink;
  }

  if (sourceNode.data.nodeType === "subgraph") {
    const port = sourceNode.data.subgraphOutputs?.find(
      p => p.name === parsed.name
    );
    if (port?.portType === "stream") return HANDLE_COLORS.video;
    if (port?.paramType) return PARAM_TYPE_COLORS[port.paramType] || "#06b6d4";
    return HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "subgraph_input") {
    const port = sourceNode.data.subgraphInputs?.find(
      p => p.name === parsed.name
    );
    if (port?.portType === "stream") return HANDLE_COLORS.video;
    if (port?.paramType) return PARAM_TYPE_COLORS[port.paramType] || "#9ca3af";
    return HANDLE_COLORS.video;
  }

  return HANDLE_COLORS.video;
}

export function buildEdgeStyle(
  sourceNode: Node<FlowNodeData> | undefined,
  sourceHandleId: string | null | undefined
): { stroke: string; strokeWidth: number } {
  const color = getEdgeColor(sourceNode, sourceHandleId);
  const parsed = parseHandleId(sourceHandleId);
  const isStreamEdge = parsed?.kind === "stream";
  const isVideoEdge =
    isStreamEdge && (parsed.name === "video" || parsed.name === "video2");
  const isBoundaryStream =
    isStreamEdge &&
    (sourceNode?.data.nodeType === "subgraph_input" ||
      sourceNode?.data.nodeType === "subgraph");
  const isVideoPathEdge =
    parsed?.kind === "param" && color === PARAM_TYPE_COLORS["video_path"];
  return {
    stroke: color,
    strokeWidth: isVideoEdge || isBoundaryStream || isVideoPathEdge ? 5 : 2,
  };
}
