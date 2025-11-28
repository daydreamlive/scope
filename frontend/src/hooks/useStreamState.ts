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
    // undefined = not loaded yet (waiting for schema)
    inputMode: undefined,
    resolution: undefined,
    seed: undefined,
    denoisingSteps: undefined,
    noiseScale: undefined,
    noiseController: undefined,
    manageCache: undefined,
    // null = explicitly disabled/not used
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
  // Note: inputMode reset is handled explicitly by handlePipelineIdChange in StreamPage
  // to allow timeline imports to preserve their imported inputMode
  useEffect(() => {
    const abortController = new AbortController();

    const fetchSchema = async () => {
      setIsLoadingSchema(true);
      try {
        const schemaResponse = await getPipelineSchema(settings.pipelineId);

        // Check if request was cancelled (pipeline changed during fetch)
        if (abortController.signal.aborted) {
          return; // Request was cancelled, new one is in flight
        }

        // Cache schema for use by other components
        cachePipelineSchema(settings.pipelineId, schemaResponse);

        // Get native mode and its config
        const nativeMode = schemaResponse.native_mode;
        const nativeModeConfig = schemaResponse.mode_configs[nativeMode];

        if (nativeModeConfig) {
          setSettings(prev => {
            // Use native mode when inputMode is undefined (pipeline just changed)
            // Otherwise preserve current mode (user explicitly selected a mode)
            const currentMode = prev.inputMode ?? nativeMode;
            const modeConfig =
              schemaResponse.mode_configs[currentMode] ?? nativeModeConfig;

            // Extract default values from JSON Schema objects
            // Force new object creation for resolution to ensure React detects the change
            const newResolution = modeConfig.resolution?.default
              ? {
                  height: modeConfig.resolution.default.height,
                  width: modeConfig.resolution.default.width,
                }
              : undefined;

            return {
              ...prev,
              inputMode: prev.inputMode ?? nativeMode, // Set to native mode if undefined
              resolution: newResolution,
              seed: modeConfig.base_seed?.default ?? prev.seed,
              denoisingSteps:
                modeConfig.denoising_steps?.default ??
                prev.denoisingSteps ??
                undefined,
              noiseScale:
                modeConfig.noise_scale?.default ?? prev.noiseScale ?? undefined,
              noiseController:
                modeConfig.noise_controller?.default ??
                prev.noiseController ??
                undefined,
              manageCache: modeConfig.manage_cache?.default ?? prev.manageCache,
              kvCacheAttentionBias:
                modeConfig.kv_cache_attention_bias?.default ??
                prev.kvCacheAttentionBias ??
                undefined,
            };
          });
        }
        setIsLoadingSchema(false);
      } catch (error) {
        // Ignore errors from cancelled requests
        if (abortController.signal.aborted) {
          return;
        }
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

    return () => {
      abortController.abort();
    };
  }, [settings.pipelineId]);

  // Set recommended quantization when krea-realtime-video is selected
  useEffect(() => {
    if (
      settings.pipelineId === "krea-realtime-video" &&
      hardwareInfo?.vram_gb != null
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
    isLoadingSchema,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
