/**
 * DynamicField - Renders a single field with conditional visibility.
 *
 * This component:
 * - Evaluates visibility conditions (ui:showIf, ui:hideInModes)
 * - Handles read-only fields
 * - Renders the appropriate control via DynamicControl
 */

import { DynamicControl } from "./dynamic-controls";
import { shouldShowField } from "../lib/conditionEvaluator";
import type {
  ResolvedSchemaProperty,
  RenderableParameter,
} from "../lib/schemaInference";
import type { InputMode } from "../types";

export interface DynamicFieldProps {
  /** Field name */
  fieldName: string;
  /** Resolved schema property with UI metadata */
  property: ResolvedSchemaProperty;
  /** Inferred control type */
  controlType: string;
  /** Display label */
  label: string;
  /** Optional tooltip */
  tooltip?: string;
  /** Current value */
  value: unknown;
  /** Change handler */
  onChange: (value: unknown) => void;
  /** Current field values (for condition evaluation) */
  fieldValues: Record<string, unknown>;
  /** Current input mode */
  inputMode?: InputMode;
  /** Whether the control is disabled */
  disabled?: boolean;
}

export function DynamicField({
  fieldName,
  property,
  controlType,
  label,
  tooltip,
  value,
  onChange,
  fieldValues,
  inputMode,
  disabled = false,
}: DynamicFieldProps) {
  // Check visibility
  const isVisible = shouldShowField(
    property.uiMetadata,
    fieldValues,
    inputMode
  );

  if (!isVisible) {
    return null;
  }

  // Check if read-only
  const isReadOnly = property.uiMetadata?.readOnly === true;

  // Create renderable parameter for DynamicControl
  const parameter: RenderableParameter = {
    name: fieldName,
    property,
    controlType: controlType as any,
    label,
    tooltip,
  };

  return (
    <div className="space-y-1">
      <DynamicControl
        parameter={parameter}
        value={value}
        onChange={(_name, val) => onChange(val)}
        disabled={disabled || isReadOnly}
      />
      {isReadOnly && (
        <p className="text-xs text-muted-foreground ml-1">
          Read-only
        </p>
      )}
    </div>
  );
}
