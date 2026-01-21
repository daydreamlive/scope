/**
 * DynamicControl - The main component for schema-driven UI rendering.
 *
 * This component inspects the JSON schema property and renders the appropriate
 * control type. It acts as a factory that maps schema types to UI components.
 *
 * The rendering logic follows these rules:
 * - `$ref` to enum definition → SelectControl
 * - `type: "boolean"` → ToggleControl
 * - `type: "number"` or `"integer"` with bounds → SliderControl
 * - `type: "number"` or `"integer"` without full bounds → NumberControl
 * - `type: "string"` → TextControl
 */

import type { RenderableParameter } from "../../lib/schemaInference";
import { SliderControl } from "./SliderControl";
import { NumberControl } from "./NumberControl";
import { ToggleControl } from "./ToggleControl";
import { SelectControl } from "./SelectControl";
import { TextControl } from "./TextControl";
import { DenoisingStepsSlider } from "../DenoisingStepsSlider";

export interface DynamicControlRendererProps {
  /** The parameter metadata including name, schema, and inferred control type */
  parameter: RenderableParameter;
  /** Current value for this parameter */
  value: unknown;
  /** Callback when value changes */
  onChange: (paramName: string, value: unknown) => void;
  /** Whether the control is disabled */
  disabled?: boolean;
}

/**
 * Renders the appropriate control based on the inferred control type.
 */
export function DynamicControl({
  parameter,
  value,
  onChange,
  disabled = false,
}: DynamicControlRendererProps) {
  const { name, property, controlType, label, tooltip } = parameter;

  const handleChange = (newValue: unknown) => {
    onChange(name, newValue);
  };

  switch (controlType) {
    case "slider":
      return (
        <SliderControl
          paramName={name}
          schema={property}
          value={(value as number) ?? property.default ?? 0}
          onChange={handleChange as (value: number) => void}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );

    case "number":
      return (
        <NumberControl
          paramName={name}
          schema={property}
          value={(value as number) ?? property.default ?? 0}
          onChange={handleChange as (value: number) => void}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );

    case "toggle":
      return (
        <ToggleControl
          paramName={name}
          schema={property}
          value={(value as boolean) ?? property.default ?? false}
          onChange={handleChange as (value: boolean) => void}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );

    case "select": {
      // Get options from resolved enum or direct enum
      const options =
        property.resolvedEnum ?? (property.enum as string[]) ?? [];
      return (
        <SelectControl
          paramName={name}
          schema={property}
          value={(value as string) ?? property.default ?? options[0] ?? ""}
          onChange={handleChange as (value: string) => void}
          options={options}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );
    }

    case "text":
      return (
        <TextControl
          paramName={name}
          schema={property}
          value={(value as string) ?? property.default ?? ""}
          onChange={handleChange as (value: string) => void}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );

    case "denoisingSteps":
      return (
        <DenoisingStepsSlider
          value={(value as number[]) ?? (property.default as number[]) ?? []}
          onChange={handleChange as (value: number[]) => void}
          disabled={disabled}
          tooltip={tooltip}
          label={label}
        />
      );

    case "textarea":
      // For now, use TextControl for textarea (can be enhanced later)
      return (
        <TextControl
          paramName={name}
          schema={property}
          value={(value as string) ?? property.default ?? ""}
          onChange={handleChange as (value: string) => void}
          label={label}
          tooltip={tooltip}
          disabled={disabled}
        />
      );

    case "unknown":
    default:
      // Fallback: show the value as read-only text
      console.warn(
        `[DynamicControl] Unknown control type for parameter "${name}"`
      );
      return (
        <div className="flex items-center justify-between gap-2 text-sm text-muted-foreground">
          <span>{label}</span>
          <span className="font-mono">{JSON.stringify(value)}</span>
        </div>
      );
  }
}
