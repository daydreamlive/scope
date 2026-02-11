import { createContext, useContext, type ReactNode } from "react";
import type { useStreamActions } from "@/hooks/useStreamActions";
import type { useStreamState } from "@/hooks/useStreamState";
import type { SettingsState } from "@/types";
import type { PipelineStatusResponse } from "@/lib/api";

type StreamActions = ReturnType<typeof useStreamActions>;
type StreamStateReturn = ReturnType<typeof useStreamState>;

export interface StreamContextValue {
  // WebRTC (from useUnifiedWebRTC — imperative, can't live in store)
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  isStreaming: boolean;
  isConnecting: boolean;
  stopStream: () => void;
  remoteStream: MediaStream | null;

  // Complex orchestration handlers (from useStreamActions)
  actions: StreamActions;

  // Pipeline info (from usePipeline)
  pipelineInfo: PipelineStatusResponse | null;

  // Derived from TanStack Query via useStreamState
  getDefaults: StreamStateReturn["getDefaults"];
  supportsNoiseControls: StreamStateReturn["supportsNoiseControls"];
  spoutAvailable: boolean;
  updateSettings: (patch: Partial<SettingsState>) => void;

  // Derived state
  isCloudMode: boolean;
  isLoading: boolean;
  isPipelineLoading: boolean;
}

const StreamContext = createContext<StreamContextValue | null>(null);

export function StreamProvider({
  value,
  children,
}: {
  value: StreamContextValue;
  children: ReactNode;
}) {
  return (
    <StreamContext.Provider value={value}>{children}</StreamContext.Provider>
  );
}

export function useStreamContext() {
  const context = useContext(StreamContext);
  if (!context) {
    throw new Error("useStreamContext must be used within StreamProvider");
  }
  return context;
}
