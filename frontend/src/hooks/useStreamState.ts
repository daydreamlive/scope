import { useState, useCallback } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
} from "../types";
import { getDefaultResolution } from "../lib/utils";

export function useStreamState() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu: 0,
    gpu: 0,
    systemRAM: 0,
    vram: 0,
    fps: 0,
    latency: 0,
  });

  const [streamStatus, setStreamStatus] = useState<StreamStatus>({
    status: "Ready",
  });

  const [settings, setSettings] = useState<SettingsState>({
    pipelineId: "streamdiffusionv2",
    resolution: getDefaultResolution("streamdiffusionv2"), // Default resolution for StreamDiffusionV2
    seed: 42,
    denoisingSteps: [700, 500], // Default for StreamDiffusionV2
    noiseScale: 0.7, // Default noise scale for StreamDiffusionV2
    noiseController: true, // Default noise controller for StreamDiffusionV2
    manageCache: true, // Default manage cache for StreamDiffusionV2
    useFp8E4m3fn: true, // Default FP8 E4M3FN for pipelines that support it
    paused: false, // Default to not paused (generating)
  });

  const [promptData, setPromptData] = useState<PromptData>({
    prompt: "",
    isProcessing: false,
  });

  const updateMetrics = useCallback((newMetrics: Partial<SystemMetrics>) => {
    setSystemMetrics(prev => ({ ...prev, ...newMetrics }));
  }, []);

  const updateStreamStatus = useCallback((newStatus: Partial<StreamStatus>) => {
    setStreamStatus(prev => ({ ...prev, ...newStatus }));
  }, []);

  const updateSettings = useCallback((newSettings: Partial<SettingsState>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const updatePrompt = useCallback((newPrompt: Partial<PromptData>) => {
    setPromptData(prev => ({ ...prev, ...newPrompt }));
  }, []);

  return {
    systemMetrics,
    streamStatus,
    settings,
    promptData,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
