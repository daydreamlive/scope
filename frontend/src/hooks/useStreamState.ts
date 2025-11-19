import { useState, useCallback, useEffect } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
} from "../types";
import { PIPELINES } from "../data/pipelines";
import { getModeDefaults } from "../lib/utils";
import { getHardwareInfo, type HardwareInfoResponse } from "../lib/api";

export function useStreamState() {
  const defaultNativeMode =
    PIPELINES["streamdiffusionv2"].nativeGenerationMode ?? "video";

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

  const streamModeDefaults = getModeDefaults(
    "streamdiffusionv2",
    defaultNativeMode
  );

  const [settings, setSettings] = useState<SettingsState>({
    pipelineId: "streamdiffusionv2",
    resolution: streamModeDefaults.resolution,
    generationMode: defaultNativeMode,
    seed: streamModeDefaults.base_seed,
    denoisingSteps: streamModeDefaults.denoising_steps,
    noiseScale: streamModeDefaults.noise_scale ?? undefined,
    noiseController: streamModeDefaults.noise_controller ?? undefined,
    manageCache: streamModeDefaults.manage_cache,
    quantization: null,
    kvCacheAttentionBias:
      streamModeDefaults.kv_cache_attention_bias ?? undefined,
    paused: false, // Default to not paused (generating)
    loraMergeStrategy: "permanent_merge", // Default LoRA merge strategy
  });

  const [promptData, setPromptData] = useState<PromptData>({
    prompt: "",
    isProcessing: false,
  });

  // Store hardware info
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfoResponse | null>(
    null
  );

  // Fetch hardware info on mount
  useEffect(() => {
    const fetchHardwareInfo = async () => {
      try {
        const info = await getHardwareInfo();
        setHardwareInfo(info);
      } catch (error) {
        console.error("Failed to fetch hardware info:", error);
      }
    };

    fetchHardwareInfo();
  }, []);

  // Set recommended quantization when krea-realtime-video is selected
  useEffect(() => {
    if (
      settings.pipelineId === "krea-realtime-video" &&
      hardwareInfo?.vram_gb !== null &&
      hardwareInfo?.vram_gb !== undefined
    ) {
      // > 40GB = no quantization (null), <= 40GB = fp8_e4m3fn
      const recommendedQuantization =
        hardwareInfo.vram_gb > 40 ? null : "fp8_e4m3fn";
      setSettings(prev => ({
        ...prev,
        quantization: recommendedQuantization,
      }));
    }
  }, [settings.pipelineId, hardwareInfo]);

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
    hardwareInfo,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
