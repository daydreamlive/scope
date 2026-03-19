import type {
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../lib/graphUtils";

export interface Blueprint {
  name: string;
  description: string;
  category: string;
  color: string;
  thumbnail: string | null;
  nodes: SerializedSubgraphNode[];
  edges: SerializedSubgraphEdge[];
}
