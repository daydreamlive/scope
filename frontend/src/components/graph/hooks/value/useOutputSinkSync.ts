import { useEffect, useRef } from "react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";

export function useOutputSinkSync(
  nodes: Node<FlowNodeData>[],
  onOutputSinkBulkChangeRef: React.RefObject<
    | ((sinks: Record<string, { enabled: boolean; name: string }>) => void)
    | undefined
  >
) {
  const prevOutputConfigsRef = useRef<string>("");

  useEffect(() => {
    const outputNodes = nodes.filter(n => n.data.nodeType === "output");
    const configs = outputNodes.map(n => ({
      type: (n.data.outputSinkType as string) || "spout",
      enabled: (n.data.outputSinkEnabled as boolean) ?? false,
      name: (n.data.outputSinkName as string) || "Scope",
    }));
    const configKey = JSON.stringify(configs);
    if (configKey === prevOutputConfigsRef.current) return;
    prevOutputConfigsRef.current = configKey;

    if (!onOutputSinkBulkChangeRef.current) return;

    const allSinkTypes = ["spout", "ndi", "syphon"];
    const sinkConfigs: Record<string, { enabled: boolean; name: string }> = {};
    for (const c of configs) {
      sinkConfigs[c.type] = { enabled: c.enabled, name: c.name };
    }
    for (const sinkType of allSinkTypes) {
      if (!sinkConfigs[sinkType]) {
        sinkConfigs[sinkType] = { enabled: false, name: "" };
      }
    }

    onOutputSinkBulkChangeRef.current(sinkConfigs);
  }, [nodes, onOutputSinkBulkChangeRef]);
}
