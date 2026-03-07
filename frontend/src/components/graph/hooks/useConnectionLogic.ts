import { useCallback } from "react";
import { addEdge, reconnectEdge } from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { getEdgeColor, PARAM_TYPE_COLORS } from "../constants";

/**
 * Resolve the effective output type of a source node. For reroute nodes,
 * walks upstream through the chain of reroutes until a concrete producer is
 * found. Returns undefined when no type can be determined.
 */
function resolveSourceType(
  node: Node<FlowNodeData>,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): "string" | "number" | "boolean" | "list_number" | undefined {
  if (visited.has(node.id)) return undefined;
  visited.add(node.id);

  const nt = node.data.nodeType;
  if (nt === "primitive") return node.data.valueType;
  if (nt === "control") {
    return node.data.controlType === "string" ? "string" : "number";
  }
  if (nt === "math") return "number";
  if (nt === "slider" || nt === "knobs" || nt === "xypad") return "number";
  if (nt === "tuple") return "list_number";
  if (nt === "reroute") {
    // Walk upstream to find source
    for (const e of edges) {
      if (e.target !== node.id) continue;
      const upstream = nodes.find(n => n.id === e.source);
      if (upstream) return resolveSourceType(upstream, nodes, edges, visited);
    }
    // Fallback to stored valueType
    return node.data.valueType;
  }
  return undefined;
}

/**
 * Determine the expected type of a target parameter port.
 */
function resolveTargetType(
  targetNode: Node<FlowNodeData>,
  targetParamName: string
): "string" | "number" | "boolean" | "list_number" | undefined {
  const nt = targetNode.data.nodeType;
  if (targetParamName === "__prompt") return "string";
  if (nt === "math") return "number";
  if (nt === "slider" || nt === "knobs" || nt === "xypad") return "number";
  if (nt === "tuple") {
    if (targetParamName === "value") return "list_number";
    if (targetParamName.startsWith("row_")) return "number";
    return undefined;
  }
  if (nt === "reroute") return undefined; // accepts any
  if (nt === "pipeline") {
    const param = targetNode.data.parameterInputs?.find(
      p => p.name === targetParamName
    );
    return param?.type;
  }
  return undefined;
}

/**
 * Walk downstream from a reroute node through other reroutes until we
 * find a typed consumer. Returns the expected type or undefined.
 */
function resolveDownstreamType(
  nodeId: string,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): "string" | "number" | "boolean" | "list_number" | undefined {
  if (visited.has(nodeId)) return undefined;
  visited.add(nodeId);

  for (const e of edges) {
    if (e.source !== nodeId) continue;
    const targetParsed = parseHandleId(e.targetHandle);
    if (!targetParsed || targetParsed.kind !== "param") continue;

    const targetNode = nodes.find(n => n.id === e.target);
    if (!targetNode) continue;

    if (targetNode.data.nodeType === "reroute") {
      // Continue downstream
      const result = resolveDownstreamType(
        targetNode.id,
        nodes,
        edges,
        visited
      );
      if (result) return result;
    } else {
      const t = resolveTargetType(targetNode, targetParsed.name);
      if (t) return t;
    }
  }
  return undefined;
}

/**
 * Collect all nodes in an upstream reroute chain (walking backward),
 * stopping when we hit a non-reroute node. Returns the chain nodes
 * and the root source node (the first non-reroute).
 */
function collectUpstreamChain(
  nodeId: string,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): { rerouteIds: string[]; rootSourceId: string | null } {
  if (visited.has(nodeId)) return { rerouteIds: [], rootSourceId: null };
  visited.add(nodeId);

  const node = nodes.find(n => n.id === nodeId);
  if (!node) return { rerouteIds: [], rootSourceId: null };

  if (node.data.nodeType !== "reroute") {
    return { rerouteIds: [], rootSourceId: node.id };
  }

  const rerouteIds = [node.id];

  // Find the upstream edge feeding into this reroute
  for (const e of edges) {
    if (e.target !== nodeId) continue;
    const upstream = collectUpstreamChain(e.source, nodes, edges, visited);
    return {
      rerouteIds: [...rerouteIds, ...upstream.rerouteIds],
      rootSourceId: upstream.rootSourceId,
    };
  }

  return { rerouteIds, rootSourceId: null };
}

export function useConnectionLogic(
  nodes: Node<FlowNodeData>[],
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>,
  handleEdgeDelete: (edgeId: string) => void
) {
  const findConnectedPipelineParams = useCallback(
    (
      sourceNodeId: string,
      edges: Edge[],
      nodes: Node<FlowNodeData>[]
    ): Array<{ nodeId: string; paramName: string }> => {
      const connected: Array<{ nodeId: string; paramName: string }> = [];

      for (const edge of edges) {
        if (edge.source !== sourceNodeId) continue;

        const targetParsed = parseHandleId(edge.targetHandle);
        if (targetParsed?.kind !== "param") continue;

        const targetNode = nodes.find(n => n.id === edge.target);
        if (targetNode?.data.nodeType !== "pipeline") continue;

        connected.push({
          nodeId: edge.target,
          paramName: targetParsed.name,
        });
      }

      return connected;
    },
    []
  );

  const isValidConnection = useCallback(
    (edgeOrConnection: Edge | Connection): boolean => {
      let connection: Connection;
      if ("source" in edgeOrConnection && "target" in edgeOrConnection) {
        connection = edgeOrConnection as Connection;
      } else {
        const edge = edgeOrConnection as Edge;
        connection = {
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle ?? null,
          targetHandle: edge.targetHandle ?? null,
        };
      }

      if (
        !connection.source ||
        !connection.target ||
        !connection.sourceHandle ||
        !connection.targetHandle
      ) {
        return true;
      }

      const sourceParsed = parseHandleId(connection.sourceHandle);
      const targetParsed = parseHandleId(connection.targetHandle);

      if (!sourceParsed || !targetParsed) {
        return true;
      }

      if (sourceParsed.kind === "stream" && targetParsed.kind === "stream") {
        return true;
      }

      if (sourceParsed.kind === "param" && targetParsed.kind === "param") {
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

        // Primitives adapt to any type
        if (sourceNode.data.nodeType === "primitive") return true;

        if (sourceNode.data.nodeType === "reroute") {
          if (!sourceNode.data.valueType) return true;
          if (targetNode.data.nodeType === "reroute") {
            return (
              !targetNode.data.valueType ||
              targetNode.data.valueType === sourceNode.data.valueType
            );
          }
        }

        if (targetNode.data.nodeType === "reroute") {
          if (!targetNode.data.valueType) return true;
          const srcType = resolveSourceType(sourceNode, nodes, []);
          if (!srcType) return true;
          return srcType === targetNode.data.valueType;
        }

        // Determine source output type
        const sourceType = resolveSourceType(sourceNode, nodes, []);

        if (!sourceType) return false;

        if (targetParsed.name === "__prompt") {
          return sourceType === "string";
        }

        // Math nodes accept number inputs
        if (targetNode.data.nodeType === "math") {
          return (
            sourceType === "number" &&
            (targetParsed.name === "a" || targetParsed.name === "b")
          );
        }

        // UI node input handles accept specific types
        if (
          targetNode.data.nodeType === "slider" ||
          targetNode.data.nodeType === "knobs" ||
          targetNode.data.nodeType === "xypad"
        ) {
          return sourceType === "number";
        }
        if (targetNode.data.nodeType === "tuple") {
          if (targetParsed.name === "value") {
            return sourceType === "list_number" || sourceType === "number";
          }
          if (targetParsed.name.startsWith("row_")) {
            return sourceType === "number";
          }
          return false;
        }

        const targetParam = targetNode.data.parameterInputs?.find(
          p => p.name === targetParsed.name
        );
        if (!targetParam) return false;

        if (targetParam.type === "list_number" && sourceType === "number") {
          return true;
        }
        if (
          targetParam.type === "list_number" &&
          sourceType === "list_number"
        ) {
          return true;
        }

        return sourceType === targetParam.type;
      }

      return false;
    },
    [nodes]
  );

  // Adapt primitive/reroute types to match connections
  const adaptNodeTypes = useCallback(
    (connection: Connection, currentEdges: Edge[]): Map<string, string> => {
      const changed = new Map<string, string>();

      const sourceNode = nodes.find(n => n.id === connection.source);
      const targetNode = nodes.find(n => n.id === connection.target);
      if (!sourceNode || !targetNode) return changed;

      const targetParsed = parseHandleId(connection.targetHandle);
      if (!targetParsed || targetParsed.kind !== "param") return changed;

      // Include new connection in edge list for traversal
      const edgesWithNew: Edge[] = [
        ...currentEdges,
        {
          id: "__pending__",
          source: connection.source ?? "",
          sourceHandle: connection.sourceHandle,
          target: connection.target ?? "",
          targetHandle: connection.targetHandle,
        },
      ];

      // Case 1: Primitive → typed target
      if (sourceNode.data.nodeType === "primitive") {
        let expectedType = resolveTargetType(targetNode, targetParsed.name);
        if (!expectedType && targetNode.data.nodeType === "reroute") {
          expectedType = resolveDownstreamType(
            targetNode.id,
            nodes,
            edgesWithNew
          );
        }
        if (
          expectedType &&
          expectedType !== "list_number" &&
          expectedType !== sourceNode.data.valueType
        ) {
          changed.set(sourceNode.id, expectedType);
          const defaultVal =
            expectedType === "boolean"
              ? false
              : expectedType === "number"
                ? 0
                : "";
          setNodes(nds =>
            nds.map(n => {
              if (n.id !== sourceNode.id) return n;
              return {
                ...n,
                data: {
                  ...n.data,
                  valueType: expectedType,
                  value: defaultVal,
                  parameterOutputs: [
                    {
                      name: "value",
                      type: expectedType,
                      defaultValue: defaultVal,
                    },
                  ],
                },
              };
            })
          );
        }
      }

      // Case 2: Something → Reroute (only from concrete producers)
      if (targetNode.data.nodeType === "reroute") {
        const isConcreteSource =
          sourceNode.data.nodeType !== "primitive" &&
          !(
            sourceNode.data.nodeType === "reroute" && !sourceNode.data.valueType
          );

        if (isConcreteSource) {
          const srcType = changed.has(sourceNode.id)
            ? (changed.get(sourceNode.id) as ReturnType<
                typeof resolveSourceType
              >)
            : resolveSourceType(sourceNode, nodes, edgesWithNew);
          if (
            srcType &&
            srcType !== "list_number" &&
            srcType !== targetNode.data.valueType
          ) {
            changed.set(targetNode.id, srcType);
            setNodes(nds =>
              nds.map(n => {
                if (n.id !== targetNode.id) return n;
                return {
                  ...n,
                  data: { ...n.data, valueType: srcType },
                };
              })
            );
          }
        }
      }

      // Case 3: Reroute → typed target (propagate backward)
      if (sourceNode.data.nodeType === "reroute") {
        let expectedType = resolveTargetType(targetNode, targetParsed.name);
        if (!expectedType && targetNode.data.nodeType === "reroute") {
          expectedType = resolveDownstreamType(
            targetNode.id,
            nodes,
            edgesWithNew
          );
        }

        if (expectedType && expectedType !== "list_number") {
          const { rerouteIds, rootSourceId } = collectUpstreamChain(
            sourceNode.id,
            nodes,
            edgesWithNew
          );

          for (const rid of rerouteIds) {
            changed.set(rid, expectedType);
          }
          if (rootSourceId) {
            const rootNode = nodes.find(n => n.id === rootSourceId);
            if (
              rootNode?.data.nodeType === "primitive" &&
              rootNode.data.valueType !== expectedType
            ) {
              changed.set(rootSourceId, expectedType);
            }
          }

          setNodes(nds =>
            nds.map(n => {
              if (rerouteIds.includes(n.id)) {
                if (n.data.valueType === expectedType) return n;
                return {
                  ...n,
                  data: { ...n.data, valueType: expectedType },
                };
              }
              if (
                rootSourceId &&
                n.id === rootSourceId &&
                n.data.nodeType === "primitive" &&
                n.data.valueType !== expectedType
              ) {
                const defaultVal =
                  expectedType === "boolean"
                    ? false
                    : expectedType === "number"
                      ? 0
                      : "";
                return {
                  ...n,
                  data: {
                    ...n.data,
                    valueType: expectedType,
                    value: defaultVal,
                    parameterOutputs: [
                      {
                        name: "value",
                        type: expectedType,
                        defaultValue: defaultVal,
                      },
                    ],
                  },
                };
              }
              return n;
            })
          );
        }
      }

      return changed;
    },
    [nodes, setNodes]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!isValidConnection(connection)) {
        return;
      }

      // Remove any existing edge to the same target handle
      let currentEdges: Edge[] = [];
      setEdges(eds => {
        currentEdges = eds.filter(
          e =>
            !(
              e.target === connection.target &&
              e.targetHandle === connection.targetHandle
            )
        );
        return currentEdges;
      });

      const changedTypes = adaptNodeTypes(connection, currentEdges);
      const sourceNode = nodes.find(n => n.id === connection.source);
      let edgeColor: string;
      const sourceChanged = changedTypes.get(connection.source ?? "");
      if (sourceChanged) {
        edgeColor = PARAM_TYPE_COLORS[sourceChanged] || "#9ca3af";
      } else {
        edgeColor = getEdgeColor(sourceNode, connection.sourceHandle);
      }

      setEdges(eds => {
        let updated = addEdge(
          {
            ...connection,
            type: "default",
            reconnectable: "target" as const,
            style: { stroke: edgeColor, strokeWidth: 2 },
            animated: false,
            data: { onDelete: handleEdgeDelete },
          },
          eds
        );

        // Refresh edge colors for changed nodes
        if (changedTypes.size > 0) {
          updated = updated.map(e => {
            const newType = changedTypes.get(e.source);
            if (newType) {
              const color = PARAM_TYPE_COLORS[newType] || "#9ca3af";
              return {
                ...e,
                style: { ...e.style, stroke: color },
              };
            }
            return e;
          });
        }

        return updated;
      });
    },
    [setEdges, nodes, handleEdgeDelete, isValidConnection, adaptNodeTypes]
  );

  const onReconnect = useCallback(
    (oldEdge: Edge, newConnection: Connection) => {
      setEdges(eds => {
        const updated = reconnectEdge(oldEdge, newConnection, eds);
        return updated.map(e => {
          if (
            e.source === newConnection.source &&
            e.target === newConnection.target &&
            e.sourceHandle === newConnection.sourceHandle &&
            e.targetHandle === newConnection.targetHandle
          ) {
            const sourceNode = nodes.find(n => n.id === e.source);
            const edgeColor = getEdgeColor(sourceNode, e.sourceHandle);
            return {
              ...e,
              type: "default",
              reconnectable: "target" as const,
              style: { stroke: edgeColor, strokeWidth: 2 },
              animated: false,
              data: { onDelete: handleEdgeDelete },
            };
          }
          return e;
        });
      });
    },
    [setEdges, nodes, handleEdgeDelete]
  );

  return {
    isValidConnection,
    onConnect,
    onReconnect,
    findConnectedPipelineParams,
  };
}
