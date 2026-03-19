import { useEffect, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { PARAM_TYPE_COLORS } from "../constants";

/**
 * Resets reroute node types when edges are removed, using shared
 * type resolution helpers. This was previously inlined in useGraphState.
 */
export function useRerouteTypeSync(
  edges: Edge[],
  nodesRef: React.RefObject<Node<FlowNodeData>[]>,
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>
) {
  const prevEdgeIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const currentEdgeIds = new Set(edges.map(e => e.id));
    const prev = prevEdgeIdsRef.current;
    prevEdgeIdsRef.current = currentEdgeIds;

    let edgesRemoved = false;
    for (const id of prev) {
      if (!currentEdgeIds.has(id)) {
        edgesRemoved = true;
        break;
      }
    }
    if (!edgesRemoved) return;

    const currentNodes = nodesRef.current;
    const rerouteNodes = currentNodes.filter(
      n => n.data.nodeType === "reroute"
    );
    if (rerouteNodes.length === 0) return;

    function findUpstreamConcreteType(
      nodeId: string,
      visited = new Set<string>()
    ): "string" | "number" | "boolean" | "list_number" | undefined {
      if (visited.has(nodeId)) return undefined;
      visited.add(nodeId);
      const inEdge = edges.find(e => e.target === nodeId);
      if (!inEdge) return undefined;
      const src = currentNodes.find(n => n.id === inEdge.source);
      if (!src) return undefined;
      if (src.data.nodeType === "reroute")
        return findUpstreamConcreteType(src.id, visited);
      if (src.data.nodeType === "primitive") return undefined;
      if (src.data.nodeType === "control")
        return src.data.controlType === "string" ? "string" : "number";
      if (src.data.nodeType === "math") return "number";
      if (
        src.data.nodeType === "slider" ||
        src.data.nodeType === "knobs" ||
        src.data.nodeType === "xypad"
      )
        return "number";
      if (src.data.nodeType === "tuple") return "list_number";
      return undefined;
    }

    function findDownstreamConcreteType(
      nodeId: string,
      visited = new Set<string>()
    ): "string" | "number" | "boolean" | "list_number" | undefined {
      if (visited.has(nodeId)) return undefined;
      visited.add(nodeId);
      for (const e of edges) {
        if (e.source !== nodeId) continue;
        const tgt = currentNodes.find(n => n.id === e.target);
        if (!tgt) continue;
        if (tgt.data.nodeType === "reroute") {
          const result = findDownstreamConcreteType(tgt.id, visited);
          if (result) return result;
          continue;
        }
        const parsed = parseHandleId(e.targetHandle);
        if (!parsed || parsed.kind !== "param") continue;
        if (tgt.data.nodeType === "math") return "number";
        if (
          tgt.data.nodeType === "slider" ||
          tgt.data.nodeType === "knobs" ||
          tgt.data.nodeType === "xypad"
        )
          return "number";
        const param = tgt.data.parameterInputs?.find(
          p => p.name === parsed.name
        );
        if (param && param.type !== "list_number") return param.type;
      }
      return undefined;
    }

    const typeUpdates = new Map<
      string,
      "string" | "number" | "boolean" | "list_number" | undefined
    >();

    for (const reroute of rerouteNodes) {
      const hasInput = edges.some(e => e.target === reroute.id);
      const hasOutput = edges.some(e => e.source === reroute.id);

      if (!hasInput && !hasOutput) {
        if (reroute.data.valueType !== undefined) {
          typeUpdates.set(reroute.id, undefined);
        }
        continue;
      }

      const downType = findDownstreamConcreteType(reroute.id);
      const upType = findUpstreamConcreteType(reroute.id);
      const determinedType = downType || upType;

      if (determinedType && determinedType !== "list_number") {
        if (reroute.data.valueType !== determinedType) {
          typeUpdates.set(reroute.id, determinedType);
        }
      } else {
        if (reroute.data.valueType !== undefined) {
          typeUpdates.set(reroute.id, undefined);
        }
      }
    }

    if (typeUpdates.size === 0) return;

    setNodes(nds =>
      nds.map(n => {
        if (!typeUpdates.has(n.id)) return n;
        const newType = typeUpdates.get(n.id);
        const validType = newType === "list_number" ? undefined : newType;
        return {
          ...n,
          data: { ...n.data, valueType: validType },
        };
      })
    );

    setEdges(eds =>
      eds.map(e => {
        if (!typeUpdates.has(e.source)) return e;
        const newType = typeUpdates.get(e.source);
        const color = newType
          ? PARAM_TYPE_COLORS[newType] || "#9ca3af"
          : "#9ca3af";
        return { ...e, style: { ...e.style, stroke: color } };
      })
    );
  }, [edges, setNodes, setEdges, nodesRef]);
}
