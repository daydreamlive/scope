import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { parseHandleId } from "../../lib/graphUtils";
import {
  PARAM_TYPE_COLORS,
  HANDLE_COLORS,
  COLOR_NUMBER,
  COLOR_STRING,
  COLOR_STREAM,
  COLOR_VACE,
  COLOR_BOOLEAN,
  COLOR_TRIGGER,
  COLOR_DEFAULT,
} from "./nodeColors";

export { PARAM_TYPE_COLORS, HANDLE_COLORS };

export function getEdgeColor(
  sourceNode: Node<FlowNodeData> | undefined,
  handleId: string | null | undefined
): string {
  if (!sourceNode || !handleId) return COLOR_DEFAULT;

  const parsed = parseHandleId(handleId);
  if (!parsed) return COLOR_DEFAULT;

  if (parsed.kind === "param") {
    if (sourceNode.data.nodeType === "primitive") {
      const valueType = sourceNode.data.valueType;
      return PARAM_TYPE_COLORS[valueType || "string"] || COLOR_DEFAULT;
    }
    if (sourceNode.data.nodeType === "reroute") {
      const valueType = sourceNode.data.valueType;
      return valueType
        ? PARAM_TYPE_COLORS[valueType] || COLOR_DEFAULT
        : COLOR_DEFAULT;
    }
    if (sourceNode.data.nodeType === "control") {
      const controlType = sourceNode.data.controlType;
      const outputType = controlType === "string" ? "string" : "number";
      return PARAM_TYPE_COLORS[outputType] || COLOR_DEFAULT;
    }
    if (sourceNode.data.nodeType === "math") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "slider") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "knobs") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "xypad") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "tuple") {
      return "#fb923c"; // orange-400 (list_number)
    }
    if (sourceNode.data.nodeType === "image") {
      return sourceNode.data.mediaType === "video"
        ? COLOR_STREAM
        : COLOR_STRING;
    }
    if (sourceNode.data.nodeType === "vace") {
      return COLOR_VACE;
    }
    if (sourceNode.data.nodeType === "midi") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "bool") {
      return COLOR_BOOLEAN;
    }
    if (sourceNode.data.nodeType === "trigger") {
      return COLOR_TRIGGER;
    }
    if (sourceNode.data.nodeType === "tempo") {
      return COLOR_NUMBER;
    }
    if (sourceNode.data.nodeType === "prompt_list") {
      return COLOR_STRING;
    }
    if (sourceNode.data.nodeType === "prompt_blend") {
      return COLOR_STRING;
    }
    if (sourceNode.data.nodeType === "subgraph") {
      const port = sourceNode.data.subgraphOutputs?.find(
        p => p.name === parsed.name
      );
      if (port?.paramType) {
        return PARAM_TYPE_COLORS[port.paramType] || COLOR_DEFAULT;
      }
      return COLOR_DEFAULT;
    }
    if (sourceNode.data.nodeType === "subgraph_input") {
      const port = sourceNode.data.subgraphInputs?.find(
        p => p.name === parsed.name
      );
      if (port?.paramType) {
        return PARAM_TYPE_COLORS[port.paramType] || COLOR_DEFAULT;
      }
      return COLOR_DEFAULT;
    }
    if (sourceNode.data.nodeType === "backend_node") {
      const schema = sourceNode.data.backendNodeSchema as
        | {
            outputs?: { name: string; type: string }[];
            dynamic_ports?: boolean;
          }
        | undefined;
      if (schema?.outputs) {
        const port = schema.outputs.find(o => o.name === parsed.name);
        if (port) {
          const typeMap: Record<string, string> = {
            trigger: "trigger",
            bool: "boolean",
            float: "number",
            int: "number",
            string: "string",
          };
          const resolved = typeMap[port.type] ?? port.type;
          return PARAM_TYPE_COLORS[resolved] || COLOR_DEFAULT;
        }
      }
      if (schema?.dynamic_ports) return COLOR_TRIGGER;
      return COLOR_DEFAULT;
    }
    return COLOR_DEFAULT;
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
    if (port?.paramType)
      return PARAM_TYPE_COLORS[port.paramType] || COLOR_DEFAULT;
    return HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "subgraph_input") {
    const port = sourceNode.data.subgraphInputs?.find(
      p => p.name === parsed.name
    );
    if (port?.portType === "stream") return HANDLE_COLORS.video;
    if (port?.paramType)
      return PARAM_TYPE_COLORS[port.paramType] || COLOR_DEFAULT;
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
