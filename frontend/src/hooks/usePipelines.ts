import { useState, useEffect } from "react";
import { getPipelineSchemas } from "../lib/api";
import type { InputMode } from "../types";

export interface PipelineInfo {
  name: string;
  about: string;
  docsUrl?: string;
  modified?: boolean;
  estimatedVram?: number;
  requiresModels?: boolean;
  defaultTemporalInterpolationMethod?: "linear" | "slerp";
  defaultTemporalInterpolationSteps?: number;
  supportsLoRA?: boolean;
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

        // Transform schemas into PipelineInfo format
        const transformed: Record<string, PipelineInfo> = {};

        // TODO: Just rely on the backend for everything
        for (const [id, schema] of Object.entries(schemas.pipelines)) {
          transformed[id] = {
            name: schema.name,
            about: schema.description,
            supportedModes: schema.supported_modes as InputMode[],
            defaultMode: schema.default_mode as InputMode,
            // Defaults for optional fields
            docsUrl: undefined,
            modified: false,
            estimatedVram: undefined,
            requiresModels: false,
            defaultTemporalInterpolationMethod: "slerp",
            defaultTemporalInterpolationSteps: 0,
            supportsLoRA: false,
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
