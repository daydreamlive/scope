import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useMemo, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import {
  customNodeInputHandleId,
  customNodeOutputHandleId,
  parseHandleId,
  stripCustomNodeDirection,
} from "../../../lib/graphUtils";
import type { NodeParamDef } from "../../../lib/api";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { getAnyValueFromNode } from "../utils/getValueFromNode";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePill,
  NodePillInput,
  NodePillSelect,
  NodePillToggle,
  collapsedHandleStyle,
} from "../ui";

const PARAM_PUSH_DEBOUNCE_MS = 100;

type CustomNodeType = Node<FlowNodeData, "custom_node">;

const PORT_COLORS: Record<string, string> = {
  audio: "#22c55e",
  video: "#eeeeee",
  number: "#38bdf8",
  string: "#fbbf24",
  boolean: "#34d399",
  trigger: "#f97316",
  latent: "#a855f7",
  model: "#f59e0b",
  vae: "#f59e0b",
  clip: "#f59e0b",
  conditioning: "#3b82f6",
  semantic_hints: "#06b6d4",
  config: "#6b7280",
  curve: "#ec4899",
  mask: "#ef4444",
  lora: "#f472b6",
};

function portColor(portType: string): string {
  return PORT_COLORS[portType] ?? "#9ca3af";
}

interface ParamWidgetProps {
  param: NodeParamDef;
  value: unknown;
  connected: boolean;
  onChange: (value: unknown) => void;
}

function ParamWidget({ param, value, connected, onChange }: ParamWidgetProps) {
  if (connected) {
    return (
      <NodePill className="opacity-50" title={String(value)}>
        {String(value) || "—"}
      </NodePill>
    );
  }
  if (param.param_type === "select" && Array.isArray(param.ui?.options)) {
    return (
      <NodePillSelect
        value={String(value)}
        onChange={onChange}
        options={(param.ui.options as string[]).map(o => ({
          value: o,
          label: o,
        }))}
      />
    );
  }
  if (param.param_type === "boolean") {
    return <NodePillToggle checked={Boolean(value)} onChange={onChange} />;
  }
  if (param.param_type === "number") {
    return (
      <NodePillInput
        type="number"
        value={Number(value)}
        min={param.ui?.min as number | undefined}
        max={param.ui?.max as number | undefined}
        step={param.ui?.step as number | undefined}
        onChange={v => onChange(Number(v))}
      />
    );
  }
  return (
    <NodePillInput type="text" value={String(value)} onChange={onChange} />
  );
}

export function CustomNode({ id, data, selected }: NodeProps<CustomNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const inputs = data.customNodeInputs ?? [];
  const outputs = data.customNodeOutputs ?? [];
  const params = data.customNodeParamDefs ?? [];

  // ComfyUI-style widget→input linkage: an input port whose name matches
  // a param renders on the same row as the param, and the wire's
  // upstream value overrides the widget's stored value.
  const paramNames = new Set(params.map(p => p.name));
  const linkedInputs = inputs.filter(p => paramNames.has(p.name));
  const unlinkedInputs = inputs.filter(p => !paramNames.has(p.name));

  const upstreamByPort = useMemo(() => {
    const map = new Map<string, unknown>();
    for (const e of edges) {
      if (e.target !== id) continue;
      const parsed = parseHandleId(e.targetHandle);
      if (!parsed || parsed.kind !== "stream") continue;
      const portName = stripCustomNodeDirection(parsed.name);
      const sourceNode = allNodes.find(n => n.id === e.source);
      const value = sourceNode
        ? getAnyValueFromNode(sourceNode, e.sourceHandle)
        : undefined;
      map.set(portName, value);
    }
    return map;
  }, [edges, allNodes, id]);

  // Per-param debounce so slider drags don't flood the data channel
  // between local widget edits and backend update_parameters calls.
  const onParamChangeRef = useRef(data.onCustomNodeParamChange);
  onParamChangeRef.current = data.onCustomNodeParamChange;
  const pushTimersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>(
    {}
  );
  useEffect(() => {
    const timers = pushTimersRef.current;
    return () => {
      for (const t of Object.values(timers)) clearTimeout(t);
    };
  }, []);
  const setParam = (name: string, value: unknown) => {
    updateData({
      customNodeParams: { ...data.customNodeParams, [name]: value },
    });
    const existing = pushTimersRef.current[name];
    if (existing) clearTimeout(existing);
    pushTimersRef.current[name] = setTimeout(() => {
      delete pushTimersRef.current[name];
      onParamChangeRef.current?.(name, value);
    }, PARAM_PUSH_DEBOUNCE_MS);
  };
  const displayName =
    data.customTitle ||
    data.customNodeDisplayName ||
    data.customNodeTypeId ||
    "Custom Node";

  const { setRowRef, rowPositions } = useHandlePositions([
    collapsed,
    unlinkedInputs.map(p => p.name).join("|"),
    outputs.map(p => p.name).join("|"),
    params.map(p => p.name).join("|"),
  ]);

  return (
    <NodeCard
      selected={selected}
      collapsed={collapsed}
      autoMinHeight={!collapsed}
    >
      <NodeHeader
        title={displayName}
        onTitleChange={t => updateData({ customTitle: t })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          {unlinkedInputs.map(p => (
            <div key={`in-${p.name}`} ref={setRowRef(`in:${p.name}`)}>
              <NodeParamRow label={p.name}>
                <NodePill>{p.name}</NodePill>
              </NodeParamRow>
            </div>
          ))}

          {outputs.map(p => (
            <div key={`out-${p.name}`} ref={setRowRef(`out:${p.name}`)}>
              <NodeParamRow label={p.name}>
                <NodePill>{p.name}</NodePill>
              </NodeParamRow>
            </div>
          ))}

          {params.map(p => {
            const connected = upstreamByPort.has(p.name);
            const val = connected
              ? (upstreamByPort.get(p.name) ?? "")
              : (data.customNodeParams?.[p.name] ?? p.default ?? "");
            return (
              <div key={`param-${p.name}`} ref={setRowRef(`param:${p.name}`)}>
                <NodeParamRow label={p.description || p.name}>
                  <ParamWidget
                    param={p}
                    value={val}
                    connected={connected}
                    onChange={v => setParam(p.name, v)}
                  />
                </NodeParamRow>
              </div>
            );
          })}
        </NodeBody>
      )}

      {unlinkedInputs.map(p => (
        <Handle
          key={`in-${p.name}`}
          type="target"
          position={Position.Left}
          id={customNodeInputHandleId(p.name)}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? collapsedHandleStyle("left")
              : {
                  backgroundColor: portColor(p.port_type),
                  top: rowPositions[`in:${p.name}`] ?? 0,
                  left: 0,
                }
          }
        />
      ))}

      {/* Linked-input handles ride on the param row, not on a row of
          their own — that's where the widget they override is rendered. */}
      {linkedInputs.map(p => (
        <Handle
          key={`in-linked-${p.name}`}
          type="target"
          position={Position.Left}
          id={customNodeInputHandleId(p.name)}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? collapsedHandleStyle("left")
              : {
                  backgroundColor: portColor(p.port_type),
                  top: rowPositions[`param:${p.name}`] ?? 0,
                  left: 0,
                }
          }
        />
      ))}

      {outputs.map(p => (
        <Handle
          key={`out-${p.name}`}
          type="source"
          position={Position.Right}
          id={customNodeOutputHandleId(p.name)}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? collapsedHandleStyle("right")
              : {
                  backgroundColor: portColor(p.port_type),
                  top: rowPositions[`out:${p.name}`] ?? 0,
                  right: 0,
                }
          }
        />
      ))}
    </NodeCard>
  );
}
