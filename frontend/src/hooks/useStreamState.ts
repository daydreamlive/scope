import { useState, useCallback, useEffect } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
} from "../types";
import {
  getHardwareInfo,
  getPipelineDefaults,
  type HardwareInfoResponse,
  type PipelineDefaultsResponse,
} from "../lib/api";
import { getCurrentModeConfig } from "../lib/utils";

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
    resolution: undefined,
    seed: undefined,
    denoisingSteps: undefined,
    noiseScale: undefined,
    noiseController: undefined,
    manageCache: undefined,
    quantization: null,
    kvCacheAttentionBias: undefined,
    paused: false,
    loraMergeStrategy: "permanent_merge",
  });

  const [promptData, setPromptData] = useState<PromptData>({
    prompt: "",
    isProcessing: false,
  });

  // Store hardware info
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfoResponse | null>(
    null
  );

  // Store pipeline defaults separately for reset functionality
  const [pipelineDefaults, setPipelineDefaults] =
    useState<PipelineDefaultsResponse | null>(null);

  // Track loading state for defaults
  const [isLoadingDefaults, setIsLoadingDefaults] = useState(true);

  // Fetch hardware info on mount
  useEffect(() => {
    const fetchHardwareInfo = async () => {
      try {
        const info = await getHardwareInfo();
        setHardwareInfo(info);
      } catch (error) {
        console.error("useStreamState: Failed to fetch hardware info:", error);
      }
    };

    fetchHardwareInfo();
  }, []);

  // Fetch and apply pipeline defaults when pipeline changes
  useEffect(() => {
    const fetchDefaults = async () => {
      setIsLoadingDefaults(true);
      try {
        const defaultsResponse = await getPipelineDefaults(settings.pipelineId);

        // Store defaults for reset functionality
        setPipelineDefaults(defaultsResponse);

        // Extract native mode config using utility function
        const defaults = getCurrentModeConfig(defaultsResponse);

        if (defaults) {
          setSettings(prev => ({
            ...prev,
            resolution: defaults.resolution,
            seed: defaults.base_seed,
            denoisingSteps: defaults.denoising_steps || [],
            noiseScale: defaults.noise_scale ?? undefined,
            noiseController: defaults.noise_controller ?? undefined,
            manageCache: defaults.manage_cache,
            kvCacheAttentionBias: defaults.kv_cache_attention_bias ?? undefined,
          }));
        }
        setIsLoadingDefaults(false);
      } catch (error) {
        console.error(
          "useStreamState: Failed to fetch pipeline defaults:",
          error
        );
        setIsLoadingDefaults(false);
        // No fallback - if backend is unreachable, streaming won't work anyway
        // User will see the error in console and UI will reflect the connection issue
      }
    };

    fetchDefaults();
  }, [settings.pipelineId]);

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
    pipelineDefaults,
    isLoadingDefaults,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
