import { createContext, useContext, type ReactNode } from "react";
import { usePipelines } from "@/hooks/usePipelines";
import type { PipelineInfo } from "@/types";

interface PipelinesContextValue {
  pipelines: Record<string, PipelineInfo> | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

const PipelinesContext = createContext<PipelinesContextValue | null>(null);

export function PipelinesProvider({ children }: { children: ReactNode }) {
  const pipelinesState = usePipelines();
  return (
    <PipelinesContext.Provider value={pipelinesState}>
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
