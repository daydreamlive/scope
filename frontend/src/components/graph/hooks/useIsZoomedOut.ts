import { useStore } from "@xyflow/react";

const DETAIL_THRESHOLD = 0.35;

// Returns true when zoom < threshold (re-renders only on boundary crossing)
export function useIsZoomedOut(): boolean {
  return useStore(s => s.transform[2] < DETAIL_THRESHOLD);
}
