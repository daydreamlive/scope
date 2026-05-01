import { useEffect, useMemo, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import type { Node } from "@xyflow/react";
import type { FlowNodeData, OscParamConfig } from "../../lib/graphUtils";
import {
  coerceDefaultForType,
  getNodeOscParams,
  pickPipelineOscParams,
  readNodeParamValue,
  slugifyForOsc,
  type OscParamDescriptor,
} from "../../lib/oscNodeParams";
import { usePipelinesContext } from "../../contexts/PipelinesContext";

interface OscConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The node being configured. */
  node: Node<FlowNodeData> | null;
  /** Update the node's data; called once on Save with the new oscConfig map. */
  onSave: (oscConfig: Record<string, OscParamConfig>) => void;
}

/**
 * Per-node Configure OSC modal.
 *
 * Lists every OSC-eligible param for the node (from the static
 * `oscNodeParams` descriptors for non-pipeline nodes, or from the live
 * pipeline schema for pipeline nodes). Each row has:
 *   - an Expose toggle
 *   - an editable OSC address (placeholder = computed default)
 *   - an editable default value (placeholder = current node value)
 *
 * No backend roundtrip until Save — local edits are staged in component
 * state. On save we hand the full `oscConfig` map back; the parent
 * commits to `node.data` via the usual `updateData()`.
 */
export function OscConfigDialog({
  open,
  onOpenChange,
  node,
  onSave,
}: OscConfigDialogProps) {
  const { pipelines } = usePipelinesContext();

  // Resolve descriptors for the node. Pipeline nodes pull from the live
  // schema; everything else uses the static per-node-type table.
  const descriptors: OscParamDescriptor[] = useMemo(() => {
    if (!node) return [];
    if (node.type === "pipeline") {
      const pid = node.data.pipelineId;
      if (!pid || !pipelines || !pipelines[pid]) return [];
      return pickPipelineOscParams(pipelines[pid].configSchema);
    }
    return getNodeOscParams(node.type);
  }, [node, pipelines]);

  const slug = useMemo(() => {
    if (!node) return "";
    return slugifyForOsc(node.data.customTitle, node.id);
  }, [node]);

  // Staged config: deep-clone the node's current oscConfig so cancel is lossless.
  const [staged, setStaged] = useState<Record<string, OscParamConfig>>({});
  useEffect(() => {
    if (!open || !node) return;
    setStaged({ ...(node.data.oscConfig ?? {}) });
  }, [open, node]);

  if (!node) return null;

  const computedAddress = (paramName: string) => `/scope/${slug}/${paramName}`;

  const updateRow = (paramName: string, patch: Partial<OscParamConfig>) => {
    setStaged(prev => {
      const existing = prev[paramName] ?? { exposed: false };
      return { ...prev, [paramName]: { ...existing, ...patch } };
    });
  };

  const handleSave = () => {
    // Coerce each row's default to the descriptor's declared type so the
    // OSC docs publish typed values (e.g. 0.85, true, [1,2,3]) instead of
    // the raw strings the modal's text inputs return.
    const descByName = new Map(descriptors.map(d => [d.name, d]));
    const cleaned: Record<string, OscParamConfig> = {};
    for (const [k, v] of Object.entries(staged)) {
      const desc = descByName.get(k);
      const coerced = desc
        ? coerceDefaultForType(v.default, desc.type)
        : v.default;
      const entry: OscParamConfig = { ...v };
      if (coerced === undefined) {
        delete entry.default;
      } else {
        entry.default = coerced;
      }
      // Drop rows where exposed is false AND no override fields, to keep
      // the saved map small.
      if (entry.exposed || entry.address || entry.default !== undefined) {
        cleaned[k] = entry;
      }
    }
    onSave(cleaned);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[640px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            Configure OSC — {node.data.customTitle || node.data.label || "Node"}
          </DialogTitle>
          <DialogDescription>
            Pick which params external OSC clients (TouchDesigner, etc.) can
            reach. Defaults are advisory metadata published in the OSC docs.
          </DialogDescription>
        </DialogHeader>

        {descriptors.length === 0 ? (
          <p className="text-sm text-muted-foreground py-6">
            This node type doesn&apos;t expose OSC-controllable params yet.
          </p>
        ) : (
          <div className="mt-3 space-y-2">
            <div className="grid grid-cols-[auto_auto_1fr_minmax(120px,160px)] gap-x-3 gap-y-2 text-xs items-center">
              <div className="font-semibold text-muted-foreground uppercase tracking-wider">
                Param
              </div>
              <div className="font-semibold text-muted-foreground uppercase tracking-wider text-center">
                Expose
              </div>
              <div className="font-semibold text-muted-foreground uppercase tracking-wider">
                Address
              </div>
              <div className="font-semibold text-muted-foreground uppercase tracking-wider">
                Default
              </div>
              {descriptors.map(d => {
                const row = staged[d.name];
                const exposed = row?.exposed ?? false;
                const currentValue = readNodeParamValue(node.data, d);
                return (
                  <ConfigRow
                    key={d.name}
                    descriptor={d}
                    row={row}
                    exposed={exposed}
                    currentValue={currentValue}
                    placeholderAddress={computedAddress(d.name)}
                    onToggle={v => updateRow(d.name, { exposed: v })}
                    onAddress={v =>
                      updateRow(d.name, {
                        address: v.trim() === "" ? undefined : v.trim(),
                      })
                    }
                    onDefault={v =>
                      updateRow(d.name, {
                        default: v === "" ? undefined : v,
                      })
                    }
                  />
                );
              })}
            </div>
          </div>
        )}

        <DialogFooter className="mt-4">
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface ConfigRowProps {
  descriptor: OscParamDescriptor;
  row: OscParamConfig | undefined;
  exposed: boolean;
  currentValue: unknown;
  placeholderAddress: string;
  onToggle: (v: boolean) => void;
  onAddress: (v: string) => void;
  onDefault: (v: string) => void;
}

function ConfigRow({
  descriptor,
  row,
  exposed,
  currentValue,
  placeholderAddress,
  onToggle,
  onAddress,
  onDefault,
}: ConfigRowProps) {
  const placeholderDefault =
    currentValue === undefined || currentValue === null
      ? ""
      : typeof currentValue === "object"
        ? JSON.stringify(currentValue)
        : String(currentValue);

  return (
    <>
      <div className="text-sm">
        <div className="font-medium">{descriptor.label}</div>
        <div className="text-[10px] text-muted-foreground font-mono">
          {descriptor.name} · {descriptor.type}
        </div>
      </div>
      <div className="flex items-center justify-center">
        <input
          type="checkbox"
          checked={exposed}
          onChange={e => onToggle(e.target.checked)}
          className="h-4 w-4"
        />
      </div>
      <Input
        type="text"
        value={row?.address ?? ""}
        placeholder={placeholderAddress}
        onChange={e => onAddress(e.target.value)}
        disabled={!exposed}
        className="h-8 text-xs font-mono"
      />
      <Input
        type="text"
        value={row?.default === undefined ? "" : String(row.default)}
        placeholder={placeholderDefault || "—"}
        onChange={e => onDefault(e.target.value)}
        disabled={!exposed}
        className="h-8 text-xs font-mono"
      />
    </>
  );
}
