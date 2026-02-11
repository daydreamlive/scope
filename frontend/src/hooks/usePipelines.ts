import { useMemo, useCallback } from "react";
import { usePipelineSchemas } from "./queries/usePipelineSchemas";
import type { InputMode, PipelineInfo } from "../types";
import type { PipelineSchemasResponse } from "../lib/api";

function transformSchemas(
  schemas: PipelineSchemasResponse
): Record<string, PipelineInfo> {
  const transformed: Record<string, PipelineInfo> = {};
  for (const [id, schema] of Object.entries(schemas.pipelines)) {
    // Check if pipeline supports controller input (has ctrl_input field in schema)
    const supportsControllerInput =
      schema.config_schema?.properties?.ctrl_input !== undefined;

    // Check if pipeline supports images input (has images field in schema)
    const supportsImages =
      schema.config_schema?.properties?.images !== undefined;

    transformed[id] = {
      name: schema.name,
      about: schema.description,
      supportedModes: schema.supported_modes as InputMode[],
      defaultMode: schema.default_mode as InputMode,
      supportsPrompts: schema.supports_prompts,
      defaultTemporalInterpolationMethod:
        schema.default_temporal_interpolation_method,
      defaultTemporalInterpolationSteps:
        schema.default_temporal_interpolation_steps,
      defaultSpatialInterpolationMethod:
        schema.default_spatial_interpolation_method,
      docsUrl: schema.docs_url ?? undefined,
      estimatedVram: schema.estimated_vram_gb ?? undefined,
      requiresModels: schema.requires_models,
      supportsLoRA: schema.supports_lora,
      supportsVACE: schema.supports_vace,
      usage: schema.usage,
      supportsCacheManagement: schema.supports_cache_management,
      supportsKvCacheBias: schema.supports_kv_cache_bias,
      supportsQuantization: schema.supports_quantization,
      minDimension: schema.min_dimension,
      recommendedQuantizationVramThreshold:
        schema.recommended_quantization_vram_threshold ?? undefined,
      modified: schema.modified,
      pluginName: schema.plugin_name ?? undefined,
      supportsControllerInput,
      supportsImages,
      configSchema: schema.config_schema,
    };
  }
  return transformed;
}

export function usePipelines() {
  const { data, isLoading, error, refetch } = usePipelineSchemas();

  const pipelines = useMemo(
    () => (data ? transformSchemas(data) : null),
    [data]
  );

  const refreshPipelines = useCallback(async () => {
    const result = await refetch();
    if (result.data) {
      return transformSchemas(result.data);
    }
    throw new Error("Failed to refresh pipelines");
  }, [refetch]);

  return {
    pipelines,
    isLoading,
    error: error
      ? error instanceof Error
        ? error.message
        : "Failed to fetch pipelines"
      : null,
    refreshPipelines,
    refetch: refreshPipelines,
  };
}
