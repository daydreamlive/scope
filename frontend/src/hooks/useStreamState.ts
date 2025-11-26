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
      const targetPipelineId = settings.pipelineId; // Capture current pipelineId
      setIsLoadingSchema(true);
      try {
        const schemaResponse = await getPipelineSchema(targetPipelineId);

        // Check if pipeline hasn't changed while we were fetching
        // (avoids race condition if user quickly switches pipelines)
        if (targetPipelineId !== settings.pipelineId) {
          console.log(
            `useStreamState: Ignoring stale schema for ${targetPipelineId}`
          );
          return;
        }

        // Cache schema for use by other components
        cachePipelineSchema(targetPipelineId, schemaResponse);

        // Store schema for reset functionality
        setPipelineSchema(schemaResponse);

        // Get native mode and its config
        const nativeMode = schemaResponse.native_mode;
        const nativeModeConfig = schemaResponse.mode_configs[nativeMode];

        if (nativeModeConfig) {
          // Extract default values from JSON Schema objects
          // Force new object creation for resolution to ensure React detects the change
          const newResolution = nativeModeConfig.resolution?.default
            ? {
                height: nativeModeConfig.resolution.default.height,
                width: nativeModeConfig.resolution.default.width,
              }
            : undefined;

          setSettings(prev => {
            // Double-check we're still on the same pipeline
            if (prev.pipelineId !== targetPipelineId) {
              console.log(
                `useStreamState: Pipeline changed during schema fetch, skipping settings update`
              );
              return prev;
            }

            return {
              ...prev,
              inputMode: nativeMode,
              resolution: newResolution,
              seed: nativeModeConfig.base_seed?.default,
              denoisingSteps:
                nativeModeConfig.denoising_steps?.default ?? undefined,
              noiseScale: nativeModeConfig.noise_scale?.default ?? undefined,
              noiseController:
                nativeModeConfig.noise_controller?.default ?? undefined,
              manageCache: nativeModeConfig.manage_cache?.default,
              kvCacheAttentionBias:
                nativeModeConfig.kv_cache_attention_bias?.default ?? undefined,
            };
          });
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
    pipelineSchema,
    isLoadingSchema,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
