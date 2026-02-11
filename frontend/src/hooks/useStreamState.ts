import { useCallback, useEffect } from "react";
import type { PipelineId, InputMode } from "../types";
import { usePipelineSchemas } from "./queries/usePipelineSchemas";
import { useHardwareInfo } from "./queries/useHardwareInfo";
import { useAppStore } from "../stores";

// Generic fallback defaults used before schemas are loaded.
// Resolution and denoising steps use conservative values.
const BASE_FALLBACK = {
  height: 320,
  width: 576,
  denoisingSteps: [1000, 750, 500, 250] as number[],
};

// Get fallback defaults for a pipeline before schemas are loaded
function getFallbackDefaults(mode?: InputMode) {
  const effectiveMode = mode ?? "text";
  const isVideoMode = effectiveMode === "video";

  return {
    height: BASE_FALLBACK.height,
    width: BASE_FALLBACK.width,
    denoisingSteps: BASE_FALLBACK.denoisingSteps,
    noiseScale: isVideoMode ? 0.7 : undefined,
    noiseController: isVideoMode ? true : undefined,
    defaultTemporalInterpolationSteps: undefined as number | undefined,
    inputMode: effectiveMode,
    quantization: undefined as "fp8_e4m3fn" | undefined,
  };
}

export function useStreamState() {
  const { data: pipelineSchemas, refetch: refetchSchemas } =
    usePipelineSchemas();
  const { data: hardwareInfo, refetch: refetchHardware } = useHardwareInfo();

  // Settings live in the Zustand store (single source of truth)
  const settings = useAppStore(s => s.settings);
  const storeUpdateSettings = useAppStore(s => s.updateSettings);

  // Helper to get defaults from schemas or fallback
  // When mode is provided, applies mode-specific overrides from mode_defaults
  const getDefaults = useCallback(
    (pipelineId: PipelineId, mode?: InputMode) => {
      const schema = pipelineSchemas?.pipelines[pipelineId];
      if (schema?.config_schema?.properties) {
        const props = schema.config_schema.properties;

        let height = (props.height?.default as number) ?? 512;
        let width = (props.width?.default as number) ?? 512;
        let denoisingSteps: number[] | undefined =
          (props.denoising_steps?.default as number[] | null) ?? undefined;
        let noiseScale: number | undefined =
          (props.noise_scale?.default as number | null) ?? undefined;
        let noiseController: boolean | undefined =
          (props.noise_controller?.default as boolean | null) ?? undefined;

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
          quantization: undefined as "fp8_e4m3fn" | undefined,
        };
      }
      return getFallbackDefaults(mode);
    },
    [pipelineSchemas]
  );

  // Check if a pipeline supports noise controls in video mode
  const supportsNoiseControls = useCallback(
    (pipelineId: PipelineId): boolean => {
      const schema = pipelineSchemas?.pipelines[pipelineId];
      if (schema?.mode_defaults?.video) {
        const noiseScale = schema.mode_defaults.video.noise_scale;
        return noiseScale !== undefined && noiseScale !== null;
      }
      return false;
    },
    [pipelineSchemas]
  );

  // Function to refresh pipeline schemas (can be called externally)
  const refreshPipelineSchemas = useCallback(async () => {
    const result = await refetchSchemas();
    const schemas = result.data;
    if (!schemas) throw new Error("Failed to refresh pipeline schemas");

    const availablePipelines = Object.keys(schemas.pipelines);
    const currentPipelineId = useAppStore.getState().settings.pipelineId;

    if (
      !availablePipelines.includes(currentPipelineId) &&
      availablePipelines.length > 0
    ) {
      const firstPipelineId = availablePipelines[0] as PipelineId;
      const firstPipelineSchema = schemas.pipelines[firstPipelineId];

      storeUpdateSettings({
        pipelineId: firstPipelineId,
        inputMode: firstPipelineSchema.default_mode,
      });
    }

    return schemas;
  }, [refetchSchemas, storeUpdateSettings]);

  // Function to refresh hardware info (can be called externally)
  const refreshHardwareInfo = useCallback(async () => {
    const result = await refetchHardware();
    if (!result.data) throw new Error("Failed to refresh hardware info");
    return result.data;
  }, [refetchHardware]);

  // When schemas first load, check if default pipeline is available
  useEffect(() => {
    if (!pipelineSchemas) return;

    const availablePipelines = Object.keys(pipelineSchemas.pipelines);
    const currentPipelineId = useAppStore.getState().settings.pipelineId;
    if (
      !availablePipelines.includes(currentPipelineId) &&
      availablePipelines.length > 0
    ) {
      const firstPipelineId = availablePipelines[0] as PipelineId;
      const firstPipelineSchema = pipelineSchemas.pipelines[firstPipelineId];

      storeUpdateSettings({
        pipelineId: firstPipelineId,
        inputMode: firstPipelineSchema.default_mode,
      });
    }
  }, [pipelineSchemas, storeUpdateSettings]);

  // Update inputMode when schemas load or pipeline changes
  useEffect(() => {
    if (pipelineSchemas) {
      const schema = pipelineSchemas.pipelines[settings.pipelineId];
      if (schema?.default_mode) {
        storeUpdateSettings({ inputMode: schema.default_mode });
      }
    }
  }, [pipelineSchemas, settings.pipelineId, storeUpdateSettings]);

  // Set recommended quantization based on pipeline schema and available VRAM
  useEffect(() => {
    const schema = pipelineSchemas?.pipelines[settings.pipelineId];
    const vramThreshold = schema?.recommended_quantization_vram_threshold;

    if (
      vramThreshold !== null &&
      vramThreshold !== undefined &&
      hardwareInfo?.vram_gb !== null &&
      hardwareInfo?.vram_gb !== undefined
    ) {
      const recommendedQuantization =
        hardwareInfo.vram_gb > vramThreshold ? null : "fp8_e4m3fn";
      storeUpdateSettings({ quantization: recommendedQuantization });
    } else {
      storeUpdateSettings({ quantization: null });
    }
  }, [settings.pipelineId, hardwareInfo, pipelineSchemas, storeUpdateSettings]);

  // Set recommended VACE enabled state based on pipeline schema and available VRAM
  useEffect(() => {
    const schema = pipelineSchemas?.pipelines[settings.pipelineId];
    const quantizationThreshold =
      schema?.recommended_quantization_vram_threshold;

    if (
      schema?.supports_vace &&
      quantizationThreshold !== null &&
      quantizationThreshold !== undefined &&
      hardwareInfo?.vram_gb !== null &&
      hardwareInfo?.vram_gb !== undefined
    ) {
      const recommendedVaceEnabled =
        hardwareInfo.vram_gb > quantizationThreshold;
      storeUpdateSettings({ vaceEnabled: recommendedVaceEnabled });
    }
  }, [settings.pipelineId, hardwareInfo, pipelineSchemas, storeUpdateSettings]);

  // Derive spoutAvailable from hardware info (server-side detection)
  const spoutAvailable = hardwareInfo?.spout_available ?? false;

  return {
    settings,
    updateSettings: storeUpdateSettings,
    getDefaults,
    supportsNoiseControls,
    spoutAvailable,
    refreshPipelineSchemas,
    refreshHardwareInfo,
  };
}
