import { createContext, useContext } from "react";

/** Maps node_id (e.g. "input", pipeline_id, or sink id) to a data URL */
export type PreviewMap = Record<string, string>;

export const DagPreviewContext = createContext<PreviewMap>({});

export function useDagPreview(nodeId: string): string | undefined {
  const previews = useContext(DagPreviewContext);
  return previews[nodeId];
}
