import { useCallback } from "react";
import { addEdge, reconnectEdge } from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { getEdgeColor } from "../constants";

export function useConnectionLogic(
  nodes: Node<FlowNodeData>[],
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

        let sourceType: "string" | "number" | "boolean" | undefined;

        if (sourceNode.data.nodeType === "value") {
          sourceType = sourceNode.data.valueType;
        } else if (sourceNode.data.nodeType === "control") {
          const controlType = sourceNode.data.controlType;
          sourceType = controlType === "string" ? "string" : "number";
        } else if (sourceNode.data.nodeType === "math") {
          sourceType = "number";
        }

        if (!sourceType) return false;

        if (targetParsed.name === "__prompt") {
          return sourceType === "string";
        }

        // Math nodes accept param:a and param:b inputs (both must be number)
        if (targetNode.data.nodeType === "math") {
          return (
            sourceType === "number" &&
            (targetParsed.name === "a" || targetParsed.name === "b")
          );
        }

        const targetParam = targetNode.data.parameterInputs?.find(
          p => p.name === targetParsed.name
        );
        if (!targetParam) return false;

        if (targetParam.type === "list_number" && sourceType === "number") {
          return true;
        }

        return sourceType === targetParam.type;
      }

      return false;
    },
    [nodes]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!isValidConnection(connection)) {
        return;
      }

      if (connection.target && connection.targetHandle) {
        setEdges(eds =>
          eds.filter(
            e =>
              !(
                e.target === connection.target &&
                e.targetHandle === connection.targetHandle
              )
          )
        );
      }

      const sourceNode = nodes.find(n => n.id === connection.source);
      const edgeColor = getEdgeColor(sourceNode, connection.sourceHandle);

      setEdges(eds =>
        addEdge(
          {
            ...connection,
            type: "default",
            reconnectable: "target" as const,
            style: { stroke: edgeColor, strokeWidth: 2 },
            animated: false,
            data: { onDelete: handleEdgeDelete },
          },
          eds
        )
      );
    },
    [setEdges, nodes, handleEdgeDelete, isValidConnection]
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

