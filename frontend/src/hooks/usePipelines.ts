import { useState, useEffect } from "react";
import { getPipelineSchemas } from "../lib/api";
import type { InputMode, PipelineInfo } from "../types";

export function usePipelines() {
  const [pipelines, setPipelines] = useState<Record<
    string,
    PipelineInfo
  > | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function fetchPipelines() {
      try {
        setIsLoading(true);
        const schemas = await getPipelineSchemas();

        if (!mounted) return;

        // Transform to camelCase for TypeScript conventions
        const transformed: Record<string, PipelineInfo> = {};
        for (const [id, schema] of Object.entries(schemas.pipelines)) {
          // Extract VAE types from JSON schema if vae_type field exists
          // Pydantic v2 represents enum fields using $ref to definitions
          let vaeTypes: string[] | undefined = undefined;
          const vaeTypeProperty = schema.config_schema?.properties?.vae_type;
          if (vaeTypeProperty?.$ref && schema.config_schema?.$defs) {
            const refPath = vaeTypeProperty.$ref;
            const defName = refPath.split("/").pop();
            const definition = schema.config_schema.$defs[defName || ""];
            if (definition && Array.isArray(definition.enum)) {
              vaeTypes = definition.enum as string[];
            }
          }

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
            vaeTypes,
            supportsControllerInput,
            supportsImages,
          };
        }

        setPipelines(transformed);
        setError(null);
      } catch (err) {
        if (!mounted) return;
        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch pipelines";
        setError(errorMessage);
        console.error("Failed to fetch pipelines:", err);
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    }

    fetchPipelines();

    return () => {
      mounted = false;
    };
  }, []);

  return { pipelines, isLoading, error };
}
