import { useMemo } from "react";
import type { PipelineSchemaInfo } from "../../lib/api";
import {
  parseConfigurationFields,
  inferPrimitiveFieldType,
  COMPLEX_COMPONENTS,
  type ParsedFieldConfig,
  type ComplexComponentName,
} from "../../lib/schemaSettings";
import {
  NodeParamRow,
  NodePill,
  NodePillSelect,
  NodePillInput,
  NodePillToggle,
  NodePillListInput,
  NodeSection,
  NODE_TOKENS,
} from "./node-ui";

interface NodeParametersPanelProps {
  pipelineId: string | null;
  nodeId: string;
  pipelineSchemas: Record<string, PipelineSchemaInfo>;
  parameterValues: Record<string, unknown>;
  onParameterChange: (nodeId: string, key: string, value: unknown) => void;
  isStreaming: boolean;
}

// Load params that remain editable during streaming
const RUNTIME_EDITABLE_LOAD_PARAMS = new Set(["vace_context_scale"]);

export function NodeParametersPanel({
  pipelineId,
  nodeId,
  pipelineSchemas,
  parameterValues,
  onParameterChange,
  isStreaming,
}: NodeParametersPanelProps) {
  const schema = pipelineId ? pipelineSchemas[pipelineId] : null;

  const { runtimeFields, loadFields } = useMemo(() => {
    if (!schema?.config_schema) {
      return { runtimeFields: [], loadFields: [] };
    }

    const allFields = parseConfigurationFields(
      schema.config_schema as import("../../lib/schemaSettings").ConfigSchemaLike,
      undefined // Show all modes in graph editor
    );

    const runtime: ParsedFieldConfig[] = [];
    const load: ParsedFieldConfig[] = [];

    for (const field of allFields) {
      if (field.ui.is_load_param) {
        load.push(field);
      } else {
        runtime.push(field);
      }
    }

    return { runtimeFields: runtime, loadFields: load };
  }, [schema]);

  if (!pipelineId || !schema) {
    return (
      <div className="p-4 text-sm text-zinc-500">
        Select a pipeline node to view its parameters.
      </div>
    );
  }

  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return "Null";
    if (typeof value === "boolean") return value ? "True" : "False";
    if (typeof value === "string") {
      if (value.length > 15) return value.substring(0, 12) + "...";
      return value;
    }
    return String(value);
  };

  const renderPrimitiveField = (
    field: ParsedFieldConfig,
    disabled: boolean
  ) => {
    let resolvedType = field.fieldType;

    // Check for array types with integer/number items
    const isArrayOfNumbers = (obj: Record<string, unknown>): boolean => {
      if (obj.type === "array" && obj.items) {
        const items = obj.items as { type?: string };
        return items.type === "integer" || items.type === "number";
      }
      return false;
    };
    const propAsRecord = field.prop as unknown as Record<string, unknown>;
    const anyOfVariants = propAsRecord.anyOf as
      | Record<string, unknown>[]
      | undefined;

    if (
      isArrayOfNumbers(propAsRecord) ||
      anyOfVariants?.some(v => isArrayOfNumbers(v))
    ) {
      const currentValue = parameterValues[field.key] ?? field.prop.default;
      const label = field.ui.label || field.key;
      return (
        <NodeParamRow key={field.key} label={label}>
          <NodePillListInput
            value={
              Array.isArray(currentValue)
                ? currentValue
                : Array.isArray(field.prop.default)
                  ? field.prop.default
                  : []
            }
            onChange={value => onParameterChange(nodeId, field.key, value)}
            disabled={disabled}
          />
        </NodeParamRow>
      );
    }

    // Try to infer primitive type for complex fields
    if (
      typeof resolvedType === "string" &&
      COMPLEX_COMPONENTS.includes(resolvedType as ComplexComponentName)
    ) {
      const inferred = inferPrimitiveFieldType(field.prop);
      if (!inferred) return null;
      resolvedType = inferred;
    }

    if (
      typeof resolvedType === "string" &&
      ["text", "number", "slider", "toggle", "enum"].includes(resolvedType)
    ) {
      const currentValue = parameterValues[field.key] ?? field.prop.default;
      const displayValue = formatValue(currentValue);
      const label = field.ui.label || field.key;

      if (resolvedType === "enum" && field.prop.enum) {
        const enumValues = Array.isArray(field.prop.enum)
          ? field.prop.enum
          : [];
        const options = enumValues.map((opt: unknown) => {
          const optValue = String(opt);
          return { value: optValue, label: formatValue(opt) };
        });
        return (
          <NodeParamRow key={field.key} label={label}>
            <NodePillSelect
              value={String(currentValue ?? "")}
              onChange={value => onParameterChange(nodeId, field.key, value)}
              disabled={disabled}
              options={options}
            />
          </NodeParamRow>
        );
      }

      if (resolvedType === "text") {
        return (
          <NodeParamRow key={field.key} label={label}>
            <NodePillInput
              type="text"
              value={String(currentValue ?? "")}
              onChange={value => onParameterChange(nodeId, field.key, value)}
              disabled={disabled}
            />
          </NodeParamRow>
        );
      }

      if (resolvedType === "number") {
        const min =
          typeof field.prop.minimum === "number"
            ? field.prop.minimum
            : undefined;
        const max =
          typeof field.prop.maximum === "number"
            ? field.prop.maximum
            : undefined;
        return (
          <NodeParamRow key={field.key} label={label}>
            <NodePillInput
              type="number"
              value={Number(currentValue ?? field.prop.default ?? 0)}
              onChange={value => onParameterChange(nodeId, field.key, value)}
              disabled={disabled}
              min={min}
              max={max}
            />
          </NodeParamRow>
        );
      }

      if (resolvedType === "toggle") {
        return (
          <NodeParamRow key={field.key} label={label}>
            <NodePillToggle
              checked={Boolean(currentValue ?? field.prop.default ?? false)}
              onChange={checked =>
                onParameterChange(nodeId, field.key, checked)
              }
              disabled={disabled}
            />
          </NodeParamRow>
        );
      }

      return (
        <NodeParamRow key={field.key} label={label}>
          <NodePill>{displayValue}</NodePill>
        </NodeParamRow>
      );
    }
    return null;
  };

  return (
    <div
      className={`flex flex-col gap-4 p-4 overflow-y-auto h-full ${NODE_TOKENS.panelBackground}`}
    >
      <div>
        <h3 className="text-sm font-semibold text-[#fafafa] mb-1">
          {schema.name}
        </h3>
        <p className={NODE_TOKENS.labelText}>Node: {nodeId}</p>
      </div>

      {/* Runtime Parameters */}
      {runtimeFields.length > 0 && (
        <NodeSection title="Properties">
          {runtimeFields.map(field => renderPrimitiveField(field, false))}
        </NodeSection>
      )}

      {/* Load Parameters */}
      {loadFields.length > 0 && (
        <NodeSection title="Model Parameters">
          {loadFields.map(field =>
            renderPrimitiveField(
              field,
              isStreaming && !RUNTIME_EDITABLE_LOAD_PARAMS.has(field.key)
            )
          )}
        </NodeSection>
      )}
    </div>
  );
}
