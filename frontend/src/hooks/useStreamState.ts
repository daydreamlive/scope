import { useState, useCallback, useEffect } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
} from "../types";
import {
  getHardwareInfo,
  getPipelineSchema,
  type HardwareInfoResponse,
  type PipelineSchema,
} from "../lib/api";
import { setPipelineSchema as cachePipelineSchema } from "../lib/utils";

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

  // Store pipeline schema separately for reset functionality
  const [pipelineSchema, setPipelineSchema] = useState<PipelineSchema | null>(
    null
  );

  // Track loading state for schema
  const [isLoadingSchema, setIsLoadingSchema] = useState(true);

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

  // Fetch and apply pipeline schema when pipeline changes
  useEffect(() => {
    const fetchSchema = async () => {
      setIsLoadingSchema(true);
      try {
        const schemaResponse = await getPipelineSchema(settings.pipelineId);

        // Cache schema for use by other components
        cachePipelineSchema(settings.pipelineId, schemaResponse);

        // Store schema for reset functionality
        setPipelineSchema(schemaResponse);

        // Get native mode and its config
        const nativeMode = schemaResponse.native_mode;
        const nativeModeConfig = schemaResponse.mode_configs[nativeMode];

        if (nativeModeConfig) {
          setSettings(prev => ({
            ...prev,
            generationMode: nativeMode,
            resolution: nativeModeConfig.resolution,
            seed: nativeModeConfig.base_seed,
            denoisingSteps: nativeModeConfig.denoising_steps ?? undefined,
            noiseScale: nativeModeConfig.noise_scale ?? undefined,
            noiseController: nativeModeConfig.noise_controller ?? undefined,
            manageCache: nativeModeConfig.manage_cache,
            kvCacheAttentionBias:
              nativeModeConfig.kv_cache_attention_bias ?? undefined,
          }));
        }
        setIsLoadingSchema(false);
      } catch (error) {
        console.error(
          "useStreamState: Failed to fetch pipeline schema:",
          error
        );
        setIsLoadingSchema(false);
        // No fallback - if backend is unreachable, streaming won't work anyway
        // User will see the error in console and UI will reflect the connection issue
      }
    };

    fetchSchema();
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
    pipelineSchema,
    isLoadingSchema,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
