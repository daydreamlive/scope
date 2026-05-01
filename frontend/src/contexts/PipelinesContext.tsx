import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { usePipelines } from "@/hooks/usePipelines";
import type { PipelineInfo } from "@/types";

interface PipelinesContextValue {
  pipelines: Record<string, PipelineInfo> | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<Record<string, PipelineInfo>>;
  refreshPipelines: () => Promise<Record<string, PipelineInfo>>;
  /** Counter that increments on every refreshPipelines call, usable as a dependency to trigger refetches elsewhere. */
  pipelinesVersion: number;
}

const PipelinesContext = createContext<PipelinesContextValue | null>(null);

export function PipelinesProvider({ children }: { children: ReactNode }) {
  const pipelinesState = usePipelines();
  const [pipelinesVersion, setPipelinesVersion] = useState(0);
  const innerRefresh = useRef(pipelinesState.refreshPipelines);
  innerRefresh.current = pipelinesState.refreshPipelines;

  const refreshPipelines = useCallback(async () => {
    try {
      const result = await innerRefresh.current();
      return result;
    } finally {
      // Always bump version so dependent effects (e.g. useGraphState) re-fetch
      // pipeline schemas, even if this particular refresh failed (e.g. cloud
      // proxy was momentarily unavailable during connection establishment).
      setPipelinesVersion(v => v + 1);
    }
  }, []);

  const value: PipelinesContextValue = {
    ...pipelinesState,
    refreshPipelines,
    refetch: refreshPipelines,
    pipelinesVersion,
  };

  return (
    <PipelinesContext.Provider value={value}>
      {children}
    </PipelinesContext.Provider>
  );
}

export function usePipelinesContext() {
  const context = useContext(PipelinesContext);
  if (!context) {
    throw new Error(
      "usePipelinesContext must be used within PipelinesProvider"
    );
  }
  return context;
}
