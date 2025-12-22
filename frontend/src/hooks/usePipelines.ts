import { useState, useEffect } from "react";
import { getPipelineSchemas } from "../lib/api";
import type { InputMode } from "../types";

export interface PipelineInfo {
  name: string;
  about: string;
  docsUrl?: string | null;
  estimatedVram?: number | null;
  requiresModels?: boolean;
  supportsPrompts?: boolean;
  defaultTemporalInterpolationMethod?: "linear" | "slerp";
  defaultTemporalInterpolationSteps?: number;
  supportsLoRA?: boolean;
  supportsVACE?: boolean;
  supportedModes: InputMode[];
  defaultMode: InputMode;
}

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
            docsUrl: schema.docs_url,
            estimatedVram: schema.estimated_vram_gb,
            requiresModels: schema.requires_models,
            supportsLoRA: schema.supports_lora,
            supportsVACE: schema.supports_vace,
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
