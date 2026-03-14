import { useCallback } from "react";
import { addEdge, reconnectEdge } from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData, SubgraphPort } from "../../../lib/graphUtils";
import { buildEdgeStyle, PARAM_TYPE_COLORS } from "../constants";
import type { ResolvedType } from "./typeResolution";
import {
  resolveSourceType,
  resolveTargetType,
  resolveDownstreamType,
  collectUpstreamChain,
} from "./typeResolution";
import { BOUNDARY_INPUT_ID, BOUNDARY_OUTPUT_ID } from "./useGraphNavigation";

export type AddSubgraphPortFn = (
  side: "input" | "output",
  port: SubgraphPort,
  setNodes: (
    updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
  ) => void
) => string | null;

export function useConnectionLogic(
  nodes: Node<FlowNodeData>[],
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>,
  handleEdgeDelete: (edgeId: string) => void,
  addSubgraphPortRef?: React.RefObject<AddSubgraphPortFn | null>
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
        if (
          targetNode?.data.nodeType !== "pipeline" &&
          targetNode?.data.nodeType !== "subgraph"
        )
          continue;

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

      if (
        (connection.source === BOUNDARY_INPUT_ID &&
          sourceParsed.name === "__add__") ||
        (connection.target === BOUNDARY_OUTPUT_ID &&
          targetParsed.name === "__add__")
      ) {
        return true;
      }

      if (sourceParsed.kind === "stream" && targetParsed.kind === "stream") {
        return true;
      }

      if (sourceParsed.kind === "param" && targetParsed.kind === "param") {
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

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

        const sourceType = resolveSourceType(sourceNode, nodes, []);

        if (!sourceType) return false;

        if (targetParsed.name === "__vace") {
          return sourceType === "vace";
        }
        if (sourceType === "vace") return targetParsed.name === "__vace";

        if (targetNode.data.nodeType === "vace") {
          if (targetParsed.name === "video") {
            if (sourceType !== "video_path") return false;
            return !(
              targetNode.data.vaceRefImage ||
              targetNode.data.vaceFirstFrame ||
              targetNode.data.vaceLastFrame
            );
          }
          if (
            ["ref_image", "first_frame", "last_frame"].includes(
              targetParsed.name
            )
          ) {
            if (sourceType !== "string") return false;
            return !targetNode.data.vaceVideo;
          }
          return false;
        }

        if (targetParsed.name === "__prompt") {
          return sourceType === "string";
        }

        if (targetNode.data.nodeType === "math") {
          return (
            sourceType === "number" &&
            (targetParsed.name === "a" || targetParsed.name === "b")
          );
        }
        if (targetNode.data.nodeType === "bool") {
          return sourceType === "number" && targetParsed.name === "input";
        }

        if (
          targetNode.data.nodeType === "control" &&
          targetNode.data.controlType === "string" &&
          targetNode.data.controlMode === "switch"
        ) {
          if (targetParsed.name.startsWith("item_")) {
            return sourceType === "number";
          }
          if (targetParsed.name.startsWith("str_")) {
            return sourceType === "string";
          }
          return false;
        }

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

        if (sourceNode.data.nodeType === "subgraph") {
          const port = sourceNode.data.subgraphOutputs?.find(
            p => p.name === sourceParsed.name
          );
          if (port) {
            if (port.paramType && targetNode.data.nodeType === "pipeline") {
              const targetParam = targetNode.data.parameterInputs?.find(
                p => p.name === targetParsed.name
              );
              return targetParam ? port.paramType === targetParam.type : false;
            }
            return true; // allow connections to reroute / primitive etc.
          }
        }

        if (targetNode.data.nodeType === "subgraph") {
          const port = targetNode.data.subgraphInputs?.find(
            p => p.name === targetParsed.name
          );
          if (port) {
            const srcType = resolveSourceType(sourceNode, nodes, []);
            if (!srcType) return true; // unknown → allow
            if (!port.paramType) return true;
            return srcType === port.paramType;
          }
        }

        if (sourceNode.data.nodeType === "subgraph_input") {
          const port = sourceNode.data.subgraphInputs?.find(
            p => p.name === sourceParsed.name
          );
          if (port) {
            if (port.paramType && targetNode.data.nodeType === "pipeline") {
              const targetParam = targetNode.data.parameterInputs?.find(
                p => p.name === targetParsed.name
              );
              return targetParam ? port.paramType === targetParam.type : false;
            }
            return true;
          }
        }

        if (targetNode.data.nodeType === "subgraph_output") {
          const port = targetNode.data.subgraphOutputs?.find(
            p => p.name === targetParsed.name
          );
          if (port) {
            const srcType = resolveSourceType(sourceNode, nodes, []);
            if (!srcType) return true;
            if (!port.paramType) return true;
            return srcType === port.paramType;
          }
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

  const adaptNodeTypes = useCallback(
    (connection: Connection, currentEdges: Edge[]): Map<string, string> => {
      const changed = new Map<string, string>();

      const sourceNode = nodes.find(n => n.id === connection.source);
      const targetNode = nodes.find(n => n.id === connection.target);
      if (!sourceNode || !targetNode) return changed;

      const targetParsed = parseHandleId(connection.targetHandle);
      if (!targetParsed || targetParsed.kind !== "param") return changed;

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

      const srcParsed = parseHandleId(connection.sourceHandle);
      const tgtParsed = parseHandleId(connection.targetHandle);

      const isAddSource =
        connection.source === BOUNDARY_INPUT_ID &&
        srcParsed?.name === "__add__";
      const isAddTarget =
        connection.target === BOUNDARY_OUTPUT_ID &&
        tgtParsed?.name === "__add__";

      let effectiveConnection = connection;

      if ((isAddSource || isAddTarget) && addSubgraphPortRef?.current) {
        const otherNode = isAddSource
          ? nodes.find(n => n.id === connection.target)
          : nodes.find(n => n.id === connection.source);
        const otherHandle = isAddSource
          ? connection.targetHandle
          : connection.sourceHandle;
        const otherParsed = parseHandleId(otherHandle);

        if (!otherNode || !otherParsed) return;

        const isStream = otherParsed.kind === "stream";
        const portType: "stream" | "param" = isStream ? "stream" : "param";
        let paramType: SubgraphPort["paramType"] = undefined;

        if (!isStream) {
          if (isAddSource) {
            const targetParam = otherNode.data.parameterInputs?.find(
              p => p.name === otherParsed.name
            );
            if (targetParam) {
              paramType = targetParam.type as SubgraphPort["paramType"];
            }
          } else {
            const srcType = resolveSourceType(otherNode, nodes, []);
            if (srcType) {
              paramType = srcType as SubgraphPort["paramType"];
            }
          }
        }

        const side = isAddSource ? "input" : "output";
        const boundaryId =
          side === "input" ? BOUNDARY_INPUT_ID : BOUNDARY_OUTPUT_ID;
        const boundaryNode = nodes.find(n => n.id === boundaryId);
        const existingPorts: SubgraphPort[] =
          (side === "input"
            ? boundaryNode?.data.subgraphInputs
            : boundaryNode?.data.subgraphOutputs) ?? [];
        const existingNames = new Set(existingPorts.map(p => p.name));
        let portName = otherParsed.name;
        if (existingNames.has(portName)) {
          let suffix = 2;
          while (existingNames.has(`${otherParsed.name}_${suffix}`)) suffix++;
          portName = `${otherParsed.name}_${suffix}`;
        }

        const newPort: SubgraphPort = {
          name: portName,
          portType,
          paramType: paramType || undefined,
          innerNodeId: isAddSource
            ? (connection.target ?? "")
            : (connection.source ?? ""),
          innerHandleId: isAddSource
            ? (connection.targetHandle ?? "")
            : (connection.sourceHandle ?? ""),
        };

        const newHandleId = addSubgraphPortRef.current(side, newPort, setNodes);
        if (!newHandleId) return;

        if (isAddSource) {
          effectiveConnection = {
            ...connection,
            sourceHandle: newHandleId,
          };
        } else {
          effectiveConnection = {
            ...connection,
            targetHandle: newHandleId,
          };
        }
      }

      const conn = effectiveConnection;

      setEdges(eds => {
        const filtered = eds.filter(
          e =>
            !(e.target === conn.target && e.targetHandle === conn.targetHandle)
        );

        const changedTypes = adaptNodeTypes(conn, filtered);
        const sourceNode = nodes.find(n => n.id === conn.source);
        const sourceChanged = changedTypes.get(conn.source ?? "");
        let style: { stroke: string; strokeWidth: number };
        if (sourceChanged) {
          const parsed = parseHandleId(conn.sourceHandle);
          const isVideoEdge =
            parsed?.kind === "stream" &&
            (parsed.name === "video" || parsed.name === "video2");
          style = {
            stroke: PARAM_TYPE_COLORS[sourceChanged] || "#9ca3af",
            strokeWidth: isVideoEdge ? 5 : 2,
          };
        } else {
          style = buildEdgeStyle(sourceNode, conn.sourceHandle);
        }

        let updated = addEdge(
          {
            ...conn,
            type: "default",
            reconnectable: "target" as const,
            style,
            animated: false,
            data: { onDelete: handleEdgeDelete },
          },
          filtered
        );

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
    [
      setEdges,
      nodes,
      handleEdgeDelete,
      isValidConnection,
      adaptNodeTypes,
      addSubgraphPortRef,
      setNodes,
    ]
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
