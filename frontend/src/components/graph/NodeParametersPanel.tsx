import { useMemo } from "react";
import type { PipelineSchemaInfo } from "../../lib/api";
import {
  parseConfigurationFields,
  type ParsedFieldConfig,
} from "../../lib/schemaSettings";
import { SchemaPrimitiveField } from "../PrimitiveFields";

interface NodeParametersPanelProps {
  /** The pipeline_id of the selected node */
  pipelineId: string | null;
  /** The node_id in the graph */
  nodeId: string;
  /** All available pipeline schemas */
  pipelineSchemas: Record<string, PipelineSchemaInfo>;
  /** Current parameter values for this node */
  parameterValues: Record<string, unknown>;
  /** Callback when a parameter value changes */
  onParameterChange: (nodeId: string, key: string, value: unknown) => void;
  /** Whether the stream is currently active */
  isStreaming: boolean;
}

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

  const renderPrimitiveField = (
    field: ParsedFieldConfig,
    disabled: boolean
  ) => {
    // Only render primitive fields (skip complex components for now)
    if (
      typeof field.fieldType === "string" &&
      ["text", "number", "slider", "toggle", "enum"].includes(field.fieldType)
    ) {
      return (
        <SchemaPrimitiveField
          key={field.key}
          fieldKey={field.key}
          prop={field.prop}
          value={parameterValues[field.key] ?? field.prop.default}
          onChange={(val: unknown) => onParameterChange(nodeId, field.key, val)}
          disabled={disabled}
          label={field.ui.label}
          fieldType={
            field.fieldType as "text" | "number" | "slider" | "toggle" | "enum"
          }
        />
      );
    }
    return null;
  };

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto h-full">
      <div>
        <h3 className="text-sm font-semibold text-zinc-200 mb-1">
          {schema.name}
        </h3>
        <p className="text-xs text-zinc-500">Node: {nodeId}</p>
      </div>

      {/* Runtime Parameters */}
      {runtimeFields.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
            Properties
          </h4>
          <div className="flex flex-col gap-3">
            {runtimeFields.map(field => renderPrimitiveField(field, false))}
          </div>
        </div>
      )}

      {/* Load Parameters */}
      {loadFields.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
            Model Parameters
          </h4>
          <div className="flex flex-col gap-3">
            {loadFields.map(field => renderPrimitiveField(field, isStreaming))}
          </div>
        </div>
      )}
    </div>
  );
}
