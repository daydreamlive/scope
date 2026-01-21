/**
 * DynamicSection - Renders a category section with grouped fields.
 *
 * This component:
 * - Groups fields by category
 * - Applies category config for section presentation
 * - Supports collapsible sections
 * - Evaluates section-level visibility conditions
 */

import { DynamicField } from "./DynamicField";
import { shouldShowCategory } from "../config/categoryConfig";
import { evaluateCondition } from "../lib/conditionEvaluator";
import type { ResolvedSchemaProperty } from "../lib/schemaInference";
import type { InputMode } from "../types";
import type { PipelineSchemaInfo } from "../lib/api";

export interface FieldInfo {
  name: string;
  property: ResolvedSchemaProperty;
  controlType: string;
  label: string;
  tooltip?: string;
  order: number;
}

export interface DynamicSectionProps {
  /** Category name */
  categoryName: string;
  /** Fields in this category */
  fields: FieldInfo[];
  /** Current field values (for condition evaluation) */
  fieldValues: Record<string, unknown>;
  /** Current input mode */
  inputMode?: InputMode;
  /** Change handler for field values */
  onFieldChange: (fieldName: string, value: unknown) => void;
  /** Whether controls are disabled */
  disabled?: boolean;
  /** Optional special control to render at the top of the section */
  specialControl?: React.ReactNode;
  /** Pipeline schema info (for category config) */
  schema?: PipelineSchemaInfo;
}

export function DynamicSection({
  categoryName,
  fields,
  fieldValues,
  inputMode,
  onFieldChange,
  disabled = false,
  specialControl,
}: DynamicSectionProps) {
  // Check section visibility
  const isVisible = shouldShowCategory(
    categoryName,
    fieldValues,
    inputMode,
    evaluateCondition
  );

  if (!isVisible) {
    return null;
  }

  // If no fields and no special control, don't render
  // BUT: if we have a special control, always show the section even if no fields
  if (fields.length === 0 && !specialControl) {
    return null;
  }

  // If we have a special control but no fields, still show the section
  // (special controls like VACE toggle, LoRA manager should always be visible)

  // Sort fields by order
  const sortedFields = [...fields].sort((a, b) => a.order - b.order);

  // Render fields directly without card wrapper
  return (
    <div className="space-y-4">
      {specialControl && <div>{specialControl}</div>}
      {sortedFields.map(field => (
        <DynamicField
          key={field.name}
          fieldName={field.name}
          property={field.property}
          controlType={field.controlType}
          label={field.label}
          tooltip={field.tooltip}
          value={fieldValues[field.name]}
          onChange={value => onFieldChange(field.name, value)}
          fieldValues={fieldValues}
          inputMode={inputMode}
          disabled={disabled}
        />
      ))}
    </div>
  );
}
