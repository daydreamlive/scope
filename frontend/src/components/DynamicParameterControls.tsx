/**
 * DynamicParameterControls - Schema-driven parameter UI rendering.
 *
 * This component renders UI controls dynamically based on the pipeline's
 * JSON schema. It extracts renderable parameters from the schema and
 * generates appropriate controls (sliders, toggles, selects, etc.) based
 * on the parameter types and constraints.
 *
 * Usage:
 * ```tsx
 * <DynamicParameterControls
 *   schema={pipelineSchema.config_schema}
 *   modeDefaults={pipelineSchema.mode_defaults}
 *   inputMode={currentMode}
 *   values={currentValues}
 *   onValueChange={(paramName, value) => handleChange(paramName, value)}
 * />
 * ```
 *
 * The component automatically:
 * - Infers control types from JSON schema (slider for bounded numbers, toggle for booleans, etc.)
 * - Applies mode-specific defaults
 * - Generates labels from parameter names
 * - Shows tooltips from schema descriptions
 */

import { useDynamicParameters } from "../hooks/useDynamicParameters";
import { DynamicControl } from "./dynamic-controls";
import type { PipelineConfigSchema, ModeDefaults } from "../lib/api";
import type { InputMode } from "../types";

export interface DynamicParameterControlsProps {
  /** The pipeline config schema containing parameter definitions */
  schema: PipelineConfigSchema | undefined;
  /** Mode-specific default overrides */
  modeDefaults?: Record<string, ModeDefaults>;
  /** Current input mode (text/video) */
  inputMode?: InputMode;
  /** Current parameter values */
  values: Record<string, unknown>;
  /** Callback when a parameter value changes */
  onValueChange: (paramName: string, value: unknown) => void;
  /** Whether controls should be disabled */
  disabled?: boolean;
  /** Optional CSS class name */
  className?: string;
  /** Optional filter to include only specific parameters */
  includeOnly?: string[];
  /** Optional filter to exclude specific parameters */
  exclude?: string[];
}

export function DynamicParameterControls({
  schema,
  modeDefaults,
  inputMode,
  values,
  onValueChange,
  disabled = false,
  className = "",
  includeOnly,
  exclude,
}: DynamicParameterControlsProps) {
  const { parameters, getValue, handleChange } = useDynamicParameters({
    schema,
    modeDefaults,
    inputMode,
    values,
    onValueChange,
  });

  // Apply filters
  let filteredParameters = parameters;

  if (includeOnly && includeOnly.length > 0) {
    filteredParameters = filteredParameters.filter(p =>
      includeOnly.includes(p.name)
    );
  }

  if (exclude && exclude.length > 0) {
    filteredParameters = filteredParameters.filter(
      p => !exclude.includes(p.name)
    );
  }

  if (filteredParameters.length === 0) {
    return null;
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {filteredParameters.map(parameter => (
        <DynamicControl
          key={parameter.name}
          parameter={parameter}
          value={getValue(parameter.name)}
          onChange={handleChange}
          disabled={disabled}
        />
      ))}
    </div>
  );
}
