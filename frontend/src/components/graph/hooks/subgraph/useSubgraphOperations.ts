import { useCallback, useMemo } from "react";
import type { Node, Edge } from "@xyflow/react";
import {
  generateNodeId,
  parseHandleId,
  buildHandleId,
} from "../../../../lib/graphUtils";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../../lib/graphUtils";
import { toast } from "sonner";
import { stripNonSerializable } from "../../utils/stripNodeData";

interface UseSubgraphOperationsArgs {
  nodes: Node<FlowNodeData>[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  setSelectedNodeIds: (ids: string[]) => void;
}

export function useSubgraphOperations({
  nodes,
  setNodes,
  setEdges,
  setSelectedNodeIds,
}: UseSubgraphOperationsArgs) {
  const existingIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);

  /**
   * Create a subgraph from the currently selected nodes.
   * - Detects "dangling" edges (crossing the selection boundary) and
   *   creates SubgraphPort entries for each.
   * - Removes selected nodes, adds a new subgraph node at their centroid.
   * - Reconnects external edges to the subgraph's exposed port handles.
   */
  const createSubgraphFromSelection = useCallback(
    (
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      selectedIds: string[]
    ) => {
      if (selectedIds.length < 1) {
        toast.warning("Select at least one node to create a subgraph");
        return;
      }

      const selSet = new Set(selectedIds);
      const selectedNodes = currentNodes.filter(n => selSet.has(n.id));
      const otherNodes = currentNodes.filter(n => !selSet.has(n.id));

      // Edges fully inside the selection
      const innerEdges = currentEdges.filter(
        e => selSet.has(e.source) && selSet.has(e.target)
      );
      // Edges crossing the boundary
      const incomingEdges = currentEdges.filter(
        e => !selSet.has(e.source) && selSet.has(e.target)
      );
      const outgoingEdges = currentEdges.filter(
        e => selSet.has(e.source) && !selSet.has(e.target)
      );
      // Edges not touching the selection at all
      const externalEdges = currentEdges.filter(
        e => !selSet.has(e.source) && !selSet.has(e.target)
      );

      // Calculate centroid for the subgraph node position
      const centroidX =
        selectedNodes.reduce((sum, n) => sum + n.position.x, 0) /
        selectedNodes.length;
      const centroidY =
        selectedNodes.reduce((sum, n) => sum + n.position.y, 0) /
        selectedNodes.length;

      // Build subgraph input ports from incoming edges
      const subgraphInputs: SubgraphPort[] = [];
      const inputPortNameCounts = new Map<string, number>();
      for (const edge of incomingEdges) {
        const parsed = parseHandleId(edge.targetHandle);
        if (!parsed) continue;
        const baseName = parsed.name;
        const count = inputPortNameCounts.get(baseName) ?? 0;
        inputPortNameCounts.set(baseName, count + 1);
        const portName = count > 0 ? `${baseName}_${count}` : baseName;

        // Determine paramType from the target node's parameter inputs
        let paramType: SubgraphPort["paramType"];
        if (parsed.kind === "param") {
          const targetNode = currentNodes.find(n => n.id === edge.target);
          const pInput = targetNode?.data.parameterInputs?.find(
            p => p.name === parsed.name
          );
          paramType = pInput?.type ?? "number";
        }

        subgraphInputs.push({
          name: portName,
          portType: parsed.kind,
          paramType,
          innerNodeId: edge.target,
          innerHandleId: edge.targetHandle || "",
        });
      }

      // Build subgraph output ports from outgoing edges
      const subgraphOutputs: SubgraphPort[] = [];
      const outputPortNameCounts = new Map<string, number>();
      for (const edge of outgoingEdges) {
        const parsed = parseHandleId(edge.sourceHandle);
        if (!parsed) continue;
        const baseName = parsed.name;
        const count = outputPortNameCounts.get(baseName) ?? 0;
        outputPortNameCounts.set(baseName, count + 1);
        const portName = count > 0 ? `${baseName}_${count}` : baseName;

        // Determine paramType from the source node's parameter outputs
        let paramType: SubgraphPort["paramType"];
        if (parsed.kind === "param") {
          const sourceNode = currentNodes.find(n => n.id === edge.source);
          const pOutput = sourceNode?.data.parameterOutputs?.find(
            p => p.name === parsed.name
          );
          paramType = pOutput?.type ?? "number";
        }

        subgraphOutputs.push({
          name: portName,
          portType: parsed.kind,
          paramType,
          innerNodeId: edge.source,
          innerHandleId: edge.sourceHandle || "",
        });
      }

      // Serialize inner nodes and edges
      const serializedInnerNodes: SerializedSubgraphNode[] = selectedNodes.map(
        n => {
          const w =
            n.width ??
            n.measured?.width ??
            (typeof n.style?.width === "number" ? n.style.width : undefined);
          const h =
            n.height ??
            n.measured?.height ??
            (typeof n.style?.height === "number" ? n.style.height : undefined);
          return {
            id: n.id,
            type: n.data.nodeType || n.type || "pipeline",
            position: {
              x: n.position.x - centroidX + 200,
              y: n.position.y - centroidY + 200,
            },
            ...(w && !Number.isNaN(w) ? { width: w } : {}),
            ...(h && !Number.isNaN(h) ? { height: h } : {}),
            data: stripNonSerializable(n.data, { skipBlocklist: true }),
          };
        }
      );
      const serializedInnerEdges: SerializedSubgraphEdge[] = innerEdges.map(
        e => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle ?? null,
          target: e.target,
          targetHandle: e.targetHandle ?? null,
        })
      );

      // Create the subgraph node
      const sgId = generateNodeId("subgraph", existingIds);
      const subgraphNode: Node<FlowNodeData> = {
        id: sgId,
        type: "subgraph",
        position: { x: centroidX, y: centroidY },
        data: {
          label: "Subgraph",
          nodeType: "subgraph",
          subgraphNodes: serializedInnerNodes,
          subgraphEdges: serializedInnerEdges,
          subgraphInputs,
          subgraphOutputs,
        },
      };

      // Remap incoming edges to point to the subgraph's input ports
      const remappedIncoming: Edge[] = incomingEdges.map((edge, i) => {
        const port = subgraphInputs[i];
        return {
          ...edge,
          id: `e-${sgId}-in-${port.name}`,
          target: sgId,
          targetHandle: buildHandleId(port.portType, port.name),
        };
      });

      // Remap outgoing edges to come from the subgraph's output ports
      const remappedOutgoing: Edge[] = outgoingEdges.map((edge, i) => {
        const port = subgraphOutputs[i];
        return {
          ...edge,
          id: `e-${sgId}-out-${port.name}`,
          source: sgId,
          sourceHandle: buildHandleId(port.portType, port.name),
        };
      });

      const newNodes = [...otherNodes, subgraphNode];
      const newEdges = [
        ...externalEdges,
        ...remappedIncoming,
        ...remappedOutgoing,
      ];

      setNodes(newNodes);
      setEdges(newEdges);
      setSelectedNodeIds([sgId]);
      toast.success(
        `Created subgraph with ${selectedNodes.length} node${selectedNodes.length !== 1 ? "s" : ""}`
      );
    },
    [existingIds, setNodes, setEdges, setSelectedNodeIds]
  );

  /**
   * Unpack a subgraph node – dissolve it back into individual nodes.
   */
  const unpackSubgraph = useCallback(
    (
      nodeId: string,
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[]
    ) => {
      const sgNode = currentNodes.find(n => n.id === nodeId);
      if (!sgNode || sgNode.data.nodeType !== "subgraph") return;

      const innerNodesSerialized = sgNode.data.subgraphNodes ?? [];
      const innerEdgesSerialized = sgNode.data.subgraphEdges ?? [];
      const sgInputs = sgNode.data.subgraphInputs ?? [];
      const sgOutputs = sgNode.data.subgraphOutputs ?? [];

      // Re-position inner nodes relative to the subgraph node's position
      const restoredNodes: Node<FlowNodeData>[] = innerNodesSerialized.map(
        n => {
          const sizeProps =
            n.width != null || n.height != null
              ? {
                  width: n.width ?? undefined,
                  height: n.height ?? undefined,
                  style: {
                    width: n.width ?? undefined,
                    height: n.height ?? undefined,
                  },
                }
              : {};
          return {
            id: n.id,
            type: n.type,
            position: {
              x: n.position.x + sgNode.position.x - 200,
              y: n.position.y + sgNode.position.y - 200,
            },
            ...sizeProps,
            data: {
              ...n.data,
              label: n.data.label ?? n.id,
            } as FlowNodeData,
          };
        }
      );

      // Restore inner edges
      const restoredEdges: Edge[] = innerEdgesSerialized.map(e => ({
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle ?? undefined,
        target: e.target,
        targetHandle: e.targetHandle ?? undefined,
      }));

      // Build port lookup maps: port name → inner target/source
      const inputMap = new Map<string, SubgraphPort>();
      for (const p of sgInputs) inputMap.set(p.name, p);
      const outputMap = new Map<string, SubgraphPort>();
      for (const p of sgOutputs) outputMap.set(p.name, p);

      // Remap edges that were connected to the subgraph
      const remappedExternalEdges: Edge[] = [];
      const otherEdges: Edge[] = [];
      for (const edge of currentEdges) {
        if (edge.target === nodeId) {
          // Incoming edge → remap to inner node
          const parsed = parseHandleId(edge.targetHandle);
          const port = parsed ? inputMap.get(parsed.name) : undefined;
          if (port) {
            remappedExternalEdges.push({
              ...edge,
              id: `e-unpack-${edge.source}-${port.innerNodeId}`,
              target: port.innerNodeId,
              targetHandle: port.innerHandleId,
            });
          }
        } else if (edge.source === nodeId) {
          // Outgoing edge → remap to inner node
          const parsed = parseHandleId(edge.sourceHandle);
          const port = parsed ? outputMap.get(parsed.name) : undefined;
          if (port) {
            remappedExternalEdges.push({
              ...edge,
              id: `e-unpack-${port.innerNodeId}-${edge.target}`,
              source: port.innerNodeId,
              sourceHandle: port.innerHandleId,
            });
          }
        } else {
          otherEdges.push(edge);
        }
      }

      // Remove the subgraph node, add inner nodes
      const remainingNodes = currentNodes.filter(n => n.id !== nodeId);
      setNodes([...remainingNodes, ...restoredNodes]);
      setEdges([...otherEdges, ...restoredEdges, ...remappedExternalEdges]);

      const innerIds = restoredNodes.map(n => n.id);
      setSelectedNodeIds(innerIds);
      toast.success(
        `Unpacked subgraph into ${restoredNodes.length} node${restoredNodes.length !== 1 ? "s" : ""}`
      );
    },
    [setNodes, setEdges, setSelectedNodeIds]
  );

  return { createSubgraphFromSelection, unpackSubgraph };
}
