import { useState, useCallback, useEffect } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
  PipelineId,
  InputMode,
} from "../types";
import {
  getHardwareInfo,
  getPipelineSchemas,
  type HardwareInfoResponse,
  type PipelineSchemasResponse,
} from "../lib/api";
import { getPipelineDefaultMode } from "../data/pipelines";

// Generic fallback defaults used before schemas are loaded.
// Resolution and denoising steps use conservative values; mode-specific
// values are derived from pipelines.ts when possible.
const BASE_FALLBACK = {
  height: 512,
  width: 512,
  denoisingSteps: [750, 250] as number[],
  seed: 42,
};

// Get fallback defaults for a pipeline before schemas are loaded
// Derives mode from pipelines.ts to stay in sync with frontend definitions
function getFallbackDefaults(pipelineId: PipelineId, mode?: InputMode) {
  const effectiveMode = mode ?? getPipelineDefaultMode(pipelineId);
  const isVideoMode = effectiveMode === "video";

  // Video mode gets noise controls, text mode doesn't
  return {
    height: BASE_FALLBACK.height,
    width: BASE_FALLBACK.width,
    denoisingSteps: BASE_FALLBACK.denoisingSteps,
    noiseScale: isVideoMode ? 0.7 : undefined,
    noiseController: isVideoMode ? true : undefined,
    inputMode: effectiveMode,
    seed: BASE_FALLBACK.seed,
    quantization: undefined as "fp8_e4m3fn" | undefined,
  };
}

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

  // Store pipeline schemas from backend
  const [pipelineSchemas, setPipelineSchemas] =
    useState<PipelineSchemasResponse | null>(null);

  // Helper to get defaults from schemas or fallback
  // When mode is provided, applies mode-specific overrides from mode_defaults
  // Returns undefined instead of null for optional fields to match SettingsState types
  const getDefaults = useCallback(
    (pipelineId: PipelineId, mode?: InputMode) => {
      const schema = pipelineSchemas?.pipelines[pipelineId];
      if (schema?.config_schema?.properties) {
        const props = schema.config_schema.properties;

        // Start with base defaults from config_schema
        let height = (props.height?.default as number) ?? 512;
        let width = (props.width?.default as number) ?? 512;
        let denoisingSteps: number[] | undefined =
          (props.denoising_steps?.default as number[] | null) ?? undefined;
        let noiseScale: number | undefined =
          (props.noise_scale?.default as number | null) ?? undefined;
        let noiseController: boolean | undefined =
          (props.noise_controller?.default as boolean | null) ?? undefined;

        // Apply mode-specific overrides if mode is specified and mode_defaults exist
        const effectiveMode = mode ?? schema.default_mode;
        const modeOverrides = schema.mode_defaults?.[effectiveMode];
        if (modeOverrides) {
          if (modeOverrides.height !== undefined) height = modeOverrides.height;
          if (modeOverrides.width !== undefined) width = modeOverrides.width;
          if (modeOverrides.denoising_steps !== undefined)
            denoisingSteps = modeOverrides.denoising_steps ?? undefined;
          if (modeOverrides.noise_scale !== undefined)
            noiseScale = modeOverrides.noise_scale ?? undefined;
          if (modeOverrides.noise_controller !== undefined)
            noiseController = modeOverrides.noise_controller ?? undefined;
        }

        return {
          height,
          width,
          denoisingSteps,
          noiseScale,
          noiseController,
          inputMode: effectiveMode,
          seed: (props.base_seed?.default as number) ?? 42,
          quantization: undefined as "fp8_e4m3fn" | undefined,
        };
      }
      // Fallback to derived defaults if schemas not loaded
      // Mode is derived from pipelines.ts to stay in sync
      return getFallbackDefaults(pipelineId, mode);
    },
    [pipelineSchemas]
  );

  // Check if a pipeline supports noise controls in video mode
  // Derived from schema: if video mode has noise_scale defined, noise controls are supported
  const supportsNoiseControls = useCallback(
    (pipelineId: PipelineId): boolean => {
      const schema = pipelineSchemas?.pipelines[pipelineId];
      if (schema?.mode_defaults?.video) {
        // Check if video mode explicitly defines noise_scale (not null/undefined)
        return schema.mode_defaults.video.noise_scale !== undefined;
      }
      // Fallback: check if schema has noise_scale property at all
      if (schema?.config_schema?.properties?.noise_scale) {
        return true;
      }
      // Before schemas load, use fallback knowledge
      return (
        pipelineId === "streamdiffusionv2" ||
        pipelineId === "longlive" ||
        pipelineId === "krea-realtime-video"
      );
    },
    [pipelineSchemas]
  );

  // Get initial defaults (use fallback since schemas haven't loaded yet)
  const initialDefaults = getFallbackDefaults("streamdiffusionv2");

  const [settings, setSettings] = useState<SettingsState>({
    pipelineId: "streamdiffusionv2",
    resolution: {
      height: initialDefaults.height,
      width: initialDefaults.width,
    },
    seed: initialDefaults.seed,
    denoisingSteps: initialDefaults.denoisingSteps,
    noiseScale: initialDefaults.noiseScale,
    noiseController: initialDefaults.noiseController,
    manageCache: true,
    quantization: null,
    kvCacheAttentionBias: 0.3,
    paused: false,
    loraMergeStrategy: "permanent_merge",
    inputMode: initialDefaults.inputMode,
  });

  const [promptData, setPromptData] = useState<PromptData>({
    prompt: "",
    isProcessing: false,
  });

  // Store hardware info
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfoResponse | null>(
    null
  );

  // Fetch pipeline schemas and hardware info on mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [schemasResult, hardwareResult] = await Promise.allSettled([
          getPipelineSchemas(),
          getHardwareInfo(),
        ]);

        if (schemasResult.status === "fulfilled") {
          setPipelineSchemas(schemasResult.value);
        } else {
          console.error(
            "useStreamState: Failed to fetch pipeline schemas:",
            schemasResult.reason
          );
        }

        if (hardwareResult.status === "fulfilled") {
          setHardwareInfo(hardwareResult.value);
        } else {
          console.error(
            "useStreamState: Failed to fetch hardware info:",
            hardwareResult.reason
          );
        }
      } catch (error) {
        console.error("useStreamState: Failed to fetch initial data:", error);
      }
    };

    fetchInitialData();
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
    pipelineSchemas,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
    getDefaults,
    supportsNoiseControls,
  };
}
