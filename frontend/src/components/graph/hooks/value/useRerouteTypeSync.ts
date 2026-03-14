import { useEffect, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import { PARAM_TYPE_COLORS } from "../../constants";
import {
  resolveSourceType,
  resolveDownstreamType,
  type ResolvedType,
} from "../connection/typeResolution";

/**
 * Synchronises reroute-node `valueType` (and edge colours) whenever edges
 * are removed.  Uses the shared type-resolution helpers from typeResolution.ts.
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

    /** Narrow ResolvedType to the subset reroute nodes support. */
    const toRerouteType = (
      t: ResolvedType
    ): "string" | "number" | "boolean" | undefined => {
      if (t === "string" || t === "number" || t === "boolean") return t;
      return undefined;
    };

    const typeUpdates = new Map<
      string,
      "string" | "number" | "boolean" | undefined
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

      const downType = toRerouteType(
        resolveDownstreamType(reroute.id, currentNodes, edges)
      );
      const upType = toRerouteType(
        resolveSourceType(reroute, currentNodes, edges)
      );
      const determinedType = downType || upType;

      if (determinedType) {
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
        return {
          ...n,
          data: { ...n.data, valueType: newType },
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
