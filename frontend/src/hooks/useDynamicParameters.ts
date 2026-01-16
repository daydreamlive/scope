/**
 * Hook for managing dynamic pipeline parameters from JSON schema.
 *
 * This hook:
 * 1. Extracts renderable parameters from the pipeline schema
 * 2. Manages parameter values with defaults from schema
 * 3. Provides a unified onChange handler for parameter updates
 */

import { useMemo, useCallback } from "react";
import type { PipelineConfigSchema, ModeDefaults } from "../lib/api";
import {
  extractRenderableParameters,
  type RenderableParameter,
} from "../lib/schemaInference";
import type { InputMode } from "../types";

export interface UseDynamicParametersOptions {
  /** The pipeline config schema */
  schema: PipelineConfigSchema | undefined;
  /** Mode-specific default overrides */
  modeDefaults?: Record<string, ModeDefaults>;
  /** Current input mode */
  inputMode?: InputMode;
  /** Current parameter values (controlled) */
  values: Record<string, unknown>;
  /** Callback when a parameter value changes */
  onValueChange: (paramName: string, value: unknown) => void;
}

export interface UseDynamicParametersResult {
  /** List of parameters that can be rendered dynamically */
  parameters: RenderableParameter[];
  /** Get the current value for a parameter (with default fallback) */
  getValue: (paramName: string) => unknown;
  /** Handle parameter change */
  handleChange: (paramName: string, value: unknown) => void;
  /** Get default value for a parameter (considering mode) */
  getDefaultValue: (paramName: string) => unknown;
}

/**
 * Converts a camelCase or PascalCase key to snake_case.
 */
function toSnakeCase(key: string): string {
  return key.replace(/([A-Z])/g, "_$1").toLowerCase();
}

/**
 * Converts a snake_case key to camelCase.
 */
function toCamelCase(key: string): string {
  return key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

export function useDynamicParameters({
  schema,
  modeDefaults,
  inputMode,
  values,
  onValueChange,
}: UseDynamicParametersOptions): UseDynamicParametersResult {
  // Extract renderable parameters from schema
  const parameters = useMemo(() => {
    if (!schema) return [];
    return extractRenderableParameters(schema);
  }, [schema]);

  // Get default value for a parameter, considering mode overrides
  const getDefaultValue = useCallback(
    (paramName: string): unknown => {
      if (!schema?.properties) return undefined;

      const property = schema.properties[paramName];
      if (!property) return undefined;

      // Check for mode-specific override first
      if (inputMode && modeDefaults?.[inputMode]) {
        const modeDefault = modeDefaults[inputMode];
        // Mode defaults use snake_case, so we need to check both
        const snakeKey = toSnakeCase(paramName);
        if (snakeKey in modeDefault) {
          return modeDefault[snakeKey as keyof ModeDefaults];
        }
        if (paramName in modeDefault) {
          return modeDefault[paramName as keyof ModeDefaults];
        }
      }

      // Fall back to schema default
      return property.default;
    },
    [schema, modeDefaults, inputMode]
  );

  // Get current value with fallback to default
  const getValue = useCallback(
    (paramName: string): unknown => {
      // Check both snake_case and camelCase in values
      const snakeKey = toSnakeCase(paramName);
      const camelKey = toCamelCase(paramName);

      if (paramName in values) return values[paramName];
      if (snakeKey in values) return values[snakeKey];
      if (camelKey in values) return values[camelKey];

      return getDefaultValue(paramName);
    },
    [values, getDefaultValue]
  );

  // Handle parameter change
  const handleChange = useCallback(
    (paramName: string, value: unknown) => {
      onValueChange(paramName, value);
    },
    [onValueChange]
  );

  return {
    parameters,
    getValue,
    handleChange,
    getDefaultValue,
  };
}
