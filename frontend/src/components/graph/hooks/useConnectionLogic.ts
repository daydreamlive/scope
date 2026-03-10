import { useCallback } from "react";
import { addEdge, reconnectEdge } from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildEdgeStyle, PARAM_TYPE_COLORS } from "../constants";
import type { ResolvedType } from "./typeResolution";
import {
  resolveSourceType,
  resolveTargetType,
  resolveDownstreamType,
  collectUpstreamChain,
} from "./typeResolution";

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

        // Primitives adapt
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

        // Get source type
        const sourceType = resolveSourceType(sourceNode, nodes, []);

        if (!sourceType) return false;

        // VACE: only VACE → pipeline.__vace
        if (targetParsed.name === "__vace") {
          return sourceType === "vace";
        }
        if (sourceType === "vace") {
          // VACE only connects to __vace
          return targetParsed.name === "__vace";
        }

        // VACE inputs: images OR video (mutual exclusion)
        if (targetNode.data.nodeType === "vace") {
          if (targetParsed.name === "video") {
            // Video: only video_path, reject if images connected
            if (sourceType !== "video_path") return false;
            const hasImages = !!(
              targetNode.data.vaceRefImage ||
              targetNode.data.vaceFirstFrame ||
              targetNode.data.vaceLastFrame
            );
            return !hasImages;
          }
          // Images: only string, reject if video connected
          if (
            targetParsed.name === "ref_image" ||
            targetParsed.name === "first_frame" ||
            targetParsed.name === "last_frame"
          ) {
            if (sourceType !== "string") return false;
            const hasVideo = !!targetNode.data.vaceVideo;
            return !hasVideo;
          }
          return false;
        }

        if (targetParsed.name === "__prompt") {
          return sourceType === "string";
        }

        // Math: numbers only
        if (targetNode.data.nodeType === "math") {
          return (
            sourceType === "number" &&
            (targetParsed.name === "a" || targetParsed.name === "b")
          );
        }

        // UI nodes: specific types
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

      // Include pending connection
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

      // Primitive → typed target
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
          expectedType !== "vace" &&
          expectedType !== "video_path" &&
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
                  valueType: expectedType as "string" | "number" | "boolean",
                  value: defaultVal,
                  parameterOutputs: [
                    {
                      name: "value",
                      type: expectedType as "string" | "number" | "boolean",
                      defaultValue: defaultVal,
                    },
                  ],
                },
              };
            })
          );
        }
      }

      // Something → Reroute (concrete producers only)
      if (targetNode.data.nodeType === "reroute") {
        const isConcreteSource =
          sourceNode.data.nodeType !== "primitive" &&
          !(
            sourceNode.data.nodeType === "reroute" && !sourceNode.data.valueType
          );

        if (isConcreteSource) {
          const srcType = changed.has(sourceNode.id)
            ? (changed.get(sourceNode.id) as ResolvedType)
            : resolveSourceType(sourceNode, nodes, edgesWithNew);
          if (
            srcType &&
            srcType !== "list_number" &&
            srcType !== "vace" &&
            srcType !== "video_path" &&
            srcType !== targetNode.data.valueType
          ) {
            changed.set(targetNode.id, srcType);
            setNodes(nds =>
              nds.map(n => {
                if (n.id !== targetNode.id) return n;
                return {
                  ...n,
                  data: {
                    ...n.data,
                    valueType: srcType as "string" | "number" | "boolean",
                  },
                };
              })
            );
          }
        }
      }

      // Reroute → typed target (backward propagation)
      if (sourceNode.data.nodeType === "reroute") {
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
          expectedType !== "vace" &&
          expectedType !== "video_path"
        ) {
          const narrowType = expectedType as "string" | "number" | "boolean";
          const { rerouteIds, rootSourceId } = collectUpstreamChain(
            sourceNode.id,
            nodes,
            edgesWithNew
          );

          for (const rid of rerouteIds) {
            changed.set(rid, narrowType);
          }
          if (rootSourceId) {
            const rootNode = nodes.find(n => n.id === rootSourceId);
            if (
              rootNode?.data.nodeType === "primitive" &&
              rootNode.data.valueType !== narrowType
            ) {
              changed.set(rootSourceId, narrowType);
            }
          }

          setNodes(nds =>
            nds.map(n => {
              if (rerouteIds.includes(n.id)) {
                if (n.data.valueType === narrowType) return n;
                return {
                  ...n,
                  data: { ...n.data, valueType: narrowType },
                };
              }
              if (
                rootSourceId &&
                n.id === rootSourceId &&
                n.data.nodeType === "primitive" &&
                n.data.valueType !== narrowType
              ) {
                const defaultVal =
                  narrowType === "boolean"
                    ? false
                    : narrowType === "number"
                      ? 0
                      : "";
                return {
                  ...n,
                  data: {
                    ...n.data,
                    valueType: narrowType,
                    value: defaultVal,
                    parameterOutputs: [
                      {
                        name: "value",
                        type: narrowType,
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

      // Remove old edge, adapt types, add new edge, refresh colors
      setEdges(eds => {
        // Remove existing edge to same target handle
        const filtered = eds.filter(
          e =>
            !(
              e.target === connection.target &&
              e.targetHandle === connection.targetHandle
            )
        );

        // Adapt node types
        const changedTypes = adaptNodeTypes(connection, filtered);

        // Determine edge style
        const sourceNode = nodes.find(n => n.id === connection.source);
        const sourceChanged = changedTypes.get(connection.source ?? "");
        let style: { stroke: string; strokeWidth: number };
        if (sourceChanged) {
          const parsed = parseHandleId(connection.sourceHandle);
          const isVideoEdge =
            parsed?.kind === "stream" &&
            (parsed.name === "video" || parsed.name === "video2");
          style = {
            stroke: PARAM_TYPE_COLORS[sourceChanged] || "#9ca3af",
            strokeWidth: isVideoEdge ? 5 : 2,
          };
        } else {
          style = buildEdgeStyle(sourceNode, connection.sourceHandle);
        }

        // Add new edge
        let updated = addEdge(
          {
            ...connection,
            type: "default",
            reconnectable: "target" as const,
            style,
            animated: false,
            data: { onDelete: handleEdgeDelete },
          },
          filtered
        );

        // Refresh edge colors for changed types
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
            const style = buildEdgeStyle(sourceNode, e.sourceHandle);
            return {
              ...e,
              type: "default",
              reconnectable: "target" as const,
              style,
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
