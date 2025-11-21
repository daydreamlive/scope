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
  type PipelineDefaults,
} from "../lib/api";
import { setPipelineDefaults as cachePipelineDefaults } from "../lib/utils";

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
    generationMode: undefined,
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
    useState<PipelineDefaults | null>(null);

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

        // Cache defaults for use by other components
        cachePipelineDefaults(settings.pipelineId, defaultsResponse);

        // Store defaults for reset functionality
        setPipelineDefaults(defaultsResponse);

        // Get native mode and its defaults
        const nativeMode = defaultsResponse.native_generation_mode;
        const nativeModeDefaults = defaultsResponse.modes[nativeMode];

        if (nativeModeDefaults) {
          setSettings(prev => ({
            ...prev,
            generationMode: nativeMode,
            resolution: nativeModeDefaults.resolution,
            seed: nativeModeDefaults.base_seed,
            denoisingSteps: nativeModeDefaults.denoising_steps ?? undefined,
            noiseScale: nativeModeDefaults.noise_scale ?? undefined,
            noiseController: nativeModeDefaults.noise_controller ?? undefined,
            manageCache: nativeModeDefaults.manage_cache,
            kvCacheAttentionBias:
              nativeModeDefaults.kv_cache_attention_bias ?? undefined,
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
