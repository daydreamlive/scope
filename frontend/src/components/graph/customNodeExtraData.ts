import type { NodeDefinitionDto } from "../../lib/api";
import type { FlowNodeData } from "../../lib/graphUtils";

export function buildCustomNodeExtraData(
  def: NodeDefinitionDto
): Partial<FlowNodeData> {
  return {
    customNodeTypeId: def.node_type_id,
    customNodeDisplayName: def.display_name || def.node_type_id,
    customNodeCategory: def.category || "",
    customNodeInputs: def.inputs || [],
    customNodeOutputs: def.outputs || [],
    customNodeParamDefs: def.params || [],
    customNodeParams: Object.fromEntries(
      (def.params || [])
        .filter(p => p.default != null)
        .map(p => [p.name, p.default])
    ),
  };
}
