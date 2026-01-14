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

// Generic fallback defaults used before schemas are loaded.
// Resolution and denoising steps use conservative values.
const BASE_FALLBACK = {
  height: 320,
  width: 576,
  denoisingSteps: [1000, 750, 500, 250] as number[],
  seed: 42,
};

// Get fallback defaults for a pipeline before schemas are loaded
function getFallbackDefaults(mode?: InputMode) {
  // Default to text mode if no mode specified (will be corrected when schemas load)
  const effectiveMode = mode ?? "text";
  const isVideoMode = effectiveMode === "video";

  // Video mode gets noise controls, text mode doesn't
  return {
    height: BASE_FALLBACK.height,
    width: BASE_FALLBACK.width,
    denoisingSteps: BASE_FALLBACK.denoisingSteps,
    noiseScale: isVideoMode ? 0.7 : undefined,
    noiseController: isVideoMode ? true : undefined,
    defaultTemporalInterpolationSteps: undefined as number | undefined,
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
        let defaultTemporalInterpolationSteps =
          schema.default_temporal_interpolation_steps;
        if (modeOverrides) {
          if (modeOverrides.height !== undefined) height = modeOverrides.height;
          if (modeOverrides.width !== undefined) width = modeOverrides.width;
          if (modeOverrides.denoising_steps !== undefined)
            denoisingSteps = modeOverrides.denoising_steps ?? undefined;
          if (modeOverrides.noise_scale !== undefined)
            noiseScale = modeOverrides.noise_scale ?? undefined;
          if (modeOverrides.noise_controller !== undefined)
            noiseController = modeOverrides.noise_controller ?? undefined;
          if (modeOverrides.default_temporal_interpolation_steps !== undefined)
            defaultTemporalInterpolationSteps =
              modeOverrides.default_temporal_interpolation_steps;
        }

        return {
          height,
          width,
          denoisingSteps,
          noiseScale,
          noiseController,
          defaultTemporalInterpolationSteps,
          inputMode: effectiveMode,
          seed: (props.base_seed?.default as number) ?? 42,
          quantization: undefined as "fp8_e4m3fn" | undefined,
        };
      }
      // Fallback to derived defaults if schemas not loaded
      return getFallbackDefaults(mode);
    },
    [pipelineSchemas]
  );

  // Check if a pipeline supports noise controls in video mode
  // Derived from schema: only show if video mode explicitly defines noise_scale with a value
  const supportsNoiseControls = useCallback(
    (pipelineId: PipelineId): boolean => {
      const schema = pipelineSchemas?.pipelines[pipelineId];
      if (schema?.mode_defaults?.video) {
        // Check if video mode explicitly defines noise_scale with a non-null value
        const noiseScale = schema.mode_defaults.video.noise_scale;
        return noiseScale !== undefined && noiseScale !== null;
      }
      // If video mode doesn't define noise_scale, don't show noise controls
      return false;
    },
    [pipelineSchemas]
  );

  // Default pipeline ID to use before schemas load
  const defaultPipelineId = "longlive";

  // Get initial defaults (use fallback since schemas haven't loaded yet)
  const initialDefaults = getFallbackDefaults("text");

  const [settings, setSettings] = useState<SettingsState>({
    pipelineId: "longlive",
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
          const schemas = schemasResult.value;
          setPipelineSchemas(schemas);

          // Check if the default pipeline is available
          // If not, switch to the first available pipeline
          const availablePipelines = Object.keys(schemas.pipelines);

          if (
            !availablePipelines.includes(defaultPipelineId) &&
            availablePipelines.length > 0
          ) {
            const firstPipelineId = availablePipelines[0] as PipelineId;
            const firstPipelineSchema = schemas.pipelines[firstPipelineId];

            setSettings(prev => ({
              ...prev,
              pipelineId: firstPipelineId,
              inputMode: firstPipelineSchema.default_mode,
            }));
          }
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

  // Update inputMode when schemas load or pipeline changes
  // This sets the correct default mode for the pipeline
  useEffect(() => {
    if (pipelineSchemas) {
      const schema = pipelineSchemas.pipelines[settings.pipelineId];
      if (schema?.default_mode) {
        setSettings(prev => ({
          ...prev,
          inputMode: schema.default_mode,
        }));
      }
    }
    // Only run when schemas load or pipeline changes, NOT when inputMode changes
  }, [pipelineSchemas, settings.pipelineId]);

  // Set recommended quantization based on pipeline schema and available VRAM
  useEffect(() => {
    const schema = pipelineSchemas?.pipelines[settings.pipelineId];
    const vramThreshold = schema?.recommended_quantization_vram_threshold;

    // Only set quantization if pipeline has a recommendation and hardware info is available
    if (
      vramThreshold !== null &&
      vramThreshold !== undefined &&
      hardwareInfo?.vram_gb !== null &&
      hardwareInfo?.vram_gb !== undefined
    ) {
      // If user's VRAM > threshold, no quantization needed (null)
      // Otherwise, recommend fp8_e4m3fn quantization
      const recommendedQuantization =
        hardwareInfo.vram_gb > vramThreshold ? null : "fp8_e4m3fn";
      setSettings(prev => ({
        ...prev,
        quantization: recommendedQuantization,
      }));
    } else {
      // No recommendation from pipeline: reset quantization to null (default)
      setSettings(prev => ({
        ...prev,
        quantization: null,
      }));
    }
  }, [settings.pipelineId, hardwareInfo, pipelineSchemas]);

  // Set recommended VACE enabled state based on pipeline schema and available VRAM
  // VACE is enabled by default, but disabled if VRAM is below recommended_quantization_vram_threshold
  useEffect(() => {
    const schema = pipelineSchemas?.pipelines[settings.pipelineId];
    const quantizationThreshold =
      schema?.recommended_quantization_vram_threshold;

    // Only set vaceEnabled if pipeline supports VACE and has a quantization VRAM threshold
    if (
      schema?.supports_vace &&
      quantizationThreshold !== null &&
      quantizationThreshold !== undefined &&
      hardwareInfo?.vram_gb !== null &&
      hardwareInfo?.vram_gb !== undefined
    ) {
      // VACE is enabled by default, but disabled if VRAM is below the quantization threshold
      // (because FP8 quantization will be recommended in that case)
      const recommendedVaceEnabled =
        hardwareInfo.vram_gb > quantizationThreshold;
      setSettings(prev => ({
        ...prev,
        vaceEnabled: recommendedVaceEnabled,
      }));
    }
    // If no threshold is set, VACE remains enabled by default (from schema)
  }, [settings.pipelineId, hardwareInfo, pipelineSchemas]);

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

  // Derive spoutAvailable from hardware info (server-side detection)
  const spoutAvailable = hardwareInfo?.spout_available ?? false;

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
    spoutAvailable,
  };
}
