import { useEffect, useMemo, useRef } from "react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData, OscParamConfig } from "../../../../lib/graphUtils";
import {
  getNodeOscParams,
  readNodeParamValue,
  slugifyForOsc,
  type OscParamDescriptor,
} from "../../../../lib/oscNodeParams";
import { setOscInventory, type OscInventoryEntry } from "../../../../lib/api";
import { usePipelinesContext } from "../../../../contexts/PipelinesContext";
import type { PipelineSchemaProperty } from "../../../../lib/api";

/**
 * Map a frontend OSC type to the JSON schema string the backend expects.
 * Mirrors `OscInventoryEntry["type"]` in api.ts.
 */
function paramTypeFromSchema(prop: PipelineSchemaProperty): string {
  const t = String(prop.type ?? "any");
  if (t === "number") return "float";
  if (t === "integer") return "integer";
  if (t === "boolean") return "bool";
  return "string";
}

interface ResolvedRow {
  descriptor: OscParamDescriptor;
  config: OscParamConfig;
  /** Current value on node.data; used as default fallback for the inventory entry. */
  currentValue: unknown;
}

function resolveNodeRows(
  node: Node<FlowNodeData>,
  pipelineDescriptors: OscParamDescriptor[] | null
): ResolvedRow[] {
  const oscConfig = node.data.oscConfig ?? {};
  const descriptors =
    node.type === "pipeline"
      ? (pipelineDescriptors ?? [])
      : getNodeOscParams(node.type);

  const rows: ResolvedRow[] = [];
  for (const d of descriptors) {
    const cfg = oscConfig[d.name];
    if (!cfg?.exposed) continue;
    rows.push({
      descriptor: d,
      config: cfg,
      currentValue: readNodeParamValue(node.data, d),
    });
  }
  return rows;
}

/**
 * Derive the OSC inventory from the live graph and POST it to the
 * backend whenever it changes.
 *
 * Inventory shape per entry: see `OscInventoryEntry` in api.ts. Two
 * route paths emerge from this:
 *
 *  1. Pipeline node entries — get `pipeline_id` + `node_id` set, so the
 *     OSC server routes to the right pipeline processor.
 *  2. Non-pipeline node entries — get `node_id` only; the SSE consumer
 *     in StreamPage applies them to the matching React Flow node.
 */
export function useOscInventory(nodes: Node<FlowNodeData>[]): void {
  const { pipelines } = usePipelinesContext();

  const inventory = useMemo<OscInventoryEntry[]>(() => {
    const entries: OscInventoryEntry[] = [];

    for (const node of nodes) {
      // Resolve descriptors. Pipeline nodes need the live schema; others
      // use the static per-type table.
      let pipelineDescriptors: OscParamDescriptor[] | null = null;
      if (node.type === "pipeline") {
        const pid = node.data.pipelineId;
        if (pid && pipelines && pipelines[pid]) {
          const schema = pipelines[pid].configSchema;
          if (!schema) {
            pipelineDescriptors = [];
            continue;
          }
          pipelineDescriptors = [];
          for (const [key, rawProp] of Object.entries(schema.properties)) {
            const prop = rawProp as PipelineSchemaProperty;
            const ui = prop.ui;
            if (ui?.is_load_param !== false) continue;
            pipelineDescriptors.push({
              name: key,
              label: (ui as { label?: string } | undefined)?.label ?? key,
              type: paramTypeFromSchema(prop) as OscParamDescriptor["type"],
              min: prop.minimum as number | undefined,
              max: prop.maximum as number | undefined,
              enum: prop.enum as string[] | undefined,
              description: (prop.description as string | undefined) ?? "",
            });
          }
        }
      }

      const rows = resolveNodeRows(node, pipelineDescriptors);
      if (rows.length === 0) continue;

      const slug = slugifyForOsc(node.data.customTitle, node.id);
      const groupLabel =
        node.data.customTitle?.trim() || node.data.label || node.id;

      for (const { descriptor, config, currentValue } of rows) {
        const address =
          config.address?.trim() || `/scope/${slug}/${descriptor.name}`;
        const defaultValue =
          config.default !== undefined ? config.default : currentValue;
        const entry: OscInventoryEntry = {
          osc_address: address,
          type: descriptor.type,
          description: descriptor.description ?? "",
          node_id: node.id,
          param: descriptor.name,
          group: groupLabel,
        };
        if (descriptor.min !== undefined) entry.min = descriptor.min;
        if (descriptor.max !== undefined) entry.max = descriptor.max;
        if (descriptor.enum !== undefined) entry.enum = descriptor.enum;
        if (defaultValue !== undefined) entry.default = defaultValue;
        if (node.type === "pipeline" && node.data.pipelineId) {
          entry.pipeline_id = node.data.pipelineId;
        }
        entries.push(entry);
      }
    }

    return entries;
  }, [nodes, pipelines]);

  // Push to backend whenever the inventory hash changes. Debounce ~300ms
  // so rapid graph edits coalesce into a single POST.
  const lastSerialized = useRef<string>("");
  useEffect(() => {
    const serialized = JSON.stringify(inventory);
    if (serialized === lastSerialized.current) return;
    const handle = window.setTimeout(() => {
      lastSerialized.current = serialized;
      void setOscInventory(inventory).catch(err => {
        // Non-fatal — OSC just won't see new paths until the next push.
        console.warn("Failed to push OSC inventory:", err);
      });
    }, 300);
    return () => window.clearTimeout(handle);
  }, [inventory]);
}
