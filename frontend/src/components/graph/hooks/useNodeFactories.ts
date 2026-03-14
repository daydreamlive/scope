import { useCallback, useMemo } from "react";
import type { Node, Edge } from "@xyflow/react";
import {
  generateNodeId,
  parseHandleId,
  buildHandleId,
} from "../../../lib/graphUtils";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../lib/graphUtils";
import { toast } from "sonner";

// Node defaults

type NodeTypeKey =
  | "source"
  | "pipeline"
  | "sink"
  | "primitive"
  | "reroute"
  | "control_float"
  | "control_int"
  | "control_string"
  | "math"
  | "note"
  | "slider"
  | "knobs"
  | "xypad"
  | "tuple"
  | "output"
  | "image"
  | "vace"
  | "midi"
  | "bool"
  | "subgraph"
  | "subgraph_input"
  | "subgraph_output";

interface NodeDefaults {
  /** The React Flow node `type` */
  type: string;
  /** Prefix for `generateNodeId` */
  idPrefix: string;
  /** Default position offset x */
  defaultX: number;
  /** Fixed style, if any */
  style?: Record<string, unknown>;
  /** Initial data (merged with `{ label, nodeType }`) */
  data: Partial<FlowNodeData>;
}

const NODE_DEFAULTS: Record<NodeTypeKey, NodeDefaults> = {
  source: {
    type: "source",
    idPrefix: "input",
    defaultX: 50,
    style: { width: 240, height: 200 },
    data: { nodeType: "source" },
  },
  pipeline: {
    type: "pipeline",
    idPrefix: "pipeline",
    defaultX: 350,
    data: {
      nodeType: "pipeline",
      pipelineId: null,
      streamInputs: ["video"],
      streamOutputs: ["video"],
    },
  },
  sink: {
    type: "sink",
    idPrefix: "output",
    defaultX: 650,
    style: { width: 240, height: 200 },
    data: { nodeType: "sink" },
  },
  primitive: {
    type: "primitive",
    idPrefix: "primitive",
    defaultX: 50,
    data: {
      label: "Primitive",
      nodeType: "primitive",
      valueType: "string",
      value: "",
      parameterOutputs: [{ name: "value", type: "string", defaultValue: "" }],
    },
  },
  reroute: {
    type: "reroute",
    idPrefix: "reroute",
    defaultX: 50,
    data: { label: "Reroute", nodeType: "reroute" },
  },
  control_float: {
    type: "control",
    idPrefix: "floatControl",
    defaultX: 50,
    data: {
      label: "FloatControl",
      nodeType: "control",
      controlType: "float",
      controlPattern: "sine",
      controlSpeed: 1.0,
      controlMin: 0,
      controlMax: 1.0,
      isPlaying: false,
      parameterOutputs: [{ name: "value", type: "number", defaultValue: 0 }],
    },
  },
  control_int: {
    type: "control",
    idPrefix: "intControl",
    defaultX: 50,
    data: {
      label: "IntControl",
      nodeType: "control",
      controlType: "int",
      controlPattern: "sine",
      controlSpeed: 1.0,
      controlMin: 0,
      controlMax: 1.0,
      isPlaying: false,
      parameterOutputs: [{ name: "value", type: "number", defaultValue: 0 }],
    },
  },
  control_string: {
    type: "control",
    idPrefix: "stringControl",
    defaultX: 50,
    data: {
      label: "StringControl",
      nodeType: "control",
      controlType: "string",
      controlPattern: "sine",
      controlSpeed: 1.0,
      controlMin: 0,
      controlMax: 1.0,
      controlItems: ["item1", "item2", "item3"],
      isPlaying: false,
      parameterOutputs: [{ name: "value", type: "string", defaultValue: "" }],
    },
  },
  math: {
    type: "math",
    idPrefix: "math",
    defaultX: 50,
    data: {
      label: "Math",
      nodeType: "math",
      mathOp: "add",
      currentValue: undefined,
      parameterOutputs: [{ name: "value", type: "number", defaultValue: 0 }],
    },
  },
  note: {
    type: "note",
    idPrefix: "note",
    defaultX: 50,
    data: { label: "Note", nodeType: "note", noteText: "" },
  },
  slider: {
    type: "slider",
    idPrefix: "slider",
    defaultX: 50,
    data: {
      label: "Slider",
      nodeType: "slider",
      sliderMin: 0,
      sliderMax: 1,
      sliderStep: 0.01,
      value: 0.5,
      parameterOutputs: [{ name: "value", type: "number", defaultValue: 0.5 }],
    },
  },
  knobs: {
    type: "knobs",
    idPrefix: "knobs",
    defaultX: 50,
    data: {
      label: "Knobs",
      nodeType: "knobs",
      knobs: [
        { label: "Knob 1", min: 0, max: 1, value: 0 },
        { label: "Knob 2", min: 0, max: 1, value: 0 },
      ],
      parameterOutputs: [
        { name: "knob_0", type: "number", defaultValue: 0 },
        { name: "knob_1", type: "number", defaultValue: 0 },
      ],
    },
  },
  xypad: {
    type: "xypad",
    idPrefix: "xypad",
    defaultX: 50,
    data: {
      label: "XY Pad",
      nodeType: "xypad",
      padMinX: 0,
      padMaxX: 1,
      padMinY: 0,
      padMaxY: 1,
      padX: 0.5,
      padY: 0.5,
      parameterOutputs: [
        { name: "x", type: "number", defaultValue: 0.5 },
        { name: "y", type: "number", defaultValue: 0.5 },
      ],
    },
  },
  tuple: {
    type: "tuple",
    idPrefix: "tuple",
    defaultX: 50,
    data: {
      label: "Tuple",
      nodeType: "tuple",
      tupleValues: [999, 800, 600],
      tupleMin: 0,
      tupleMax: 1000,
      tupleStep: 1,
      tupleEnforceOrder: true,
      tupleOrderDirection: "desc",
      parameterOutputs: [
        {
          name: "value",
          type: "list_number",
          defaultValue: [999, 800, 600],
        },
      ],
    },
  },
  output: {
    type: "output",
    idPrefix: "output_sink",
    defaultX: 900,
    data: {
      label: "Output",
      nodeType: "output",
      outputSinkEnabled: false,
    },
  },
  image: {
    type: "image",
    idPrefix: "media",
    defaultX: 50,
    style: { width: 160, height: 140 },
    data: {
      label: "Media",
      nodeType: "image",
      imagePath: "",
      mediaType: "image",
      parameterOutputs: [{ name: "value", type: "string", defaultValue: "" }],
    },
  },
  vace: {
    type: "vace",
    idPrefix: "vace",
    defaultX: 50,
    style: { width: 240 },
    data: {
      label: "VACE",
      nodeType: "vace",
      vaceContextScale: 1.0,
      vaceRefImage: "",
      vaceFirstFrame: "",
      vaceLastFrame: "",
      vaceVideo: "",
      parameterOutputs: [{ name: "__vace", type: "string", defaultValue: "" }],
    },
  },
  midi: {
    type: "midi",
    idPrefix: "midi",
    defaultX: 50,
    data: {
      label: "MIDI",
      nodeType: "midi",
      midiChannels: [
        { label: "CC 1", type: "cc", channel: 0, cc: 1, value: 0 },
        { label: "CC 2", type: "cc", channel: 0, cc: 2, value: 0 },
      ],
      parameterOutputs: [
        { name: "midi_0", type: "number", defaultValue: 0 },
        { name: "midi_1", type: "number", defaultValue: 0 },
      ],
    },
  },
  bool: {
    type: "bool",
    idPrefix: "bool",
    defaultX: 50,
    data: {
      label: "Bool",
      nodeType: "bool",
      boolMode: "gate",
      boolThreshold: 0,
      value: false,
      parameterOutputs: [
        { name: "value", type: "boolean", defaultValue: false },
      ],
    },
  },
  subgraph_input: {
    type: "subgraph_input",
    idPrefix: "sg_in",
    defaultX: 50,
    data: { label: "Subgraph Inputs", nodeType: "subgraph_input" },
  },
  subgraph_output: {
    type: "subgraph_output",
    idPrefix: "sg_out",
    defaultX: 600,
    data: { label: "Subgraph Outputs", nodeType: "subgraph_output" },
  },
  subgraph: {
    type: "subgraph",
    idPrefix: "subgraph",
    defaultX: 300,
    data: {
      label: "Subgraph",
      nodeType: "subgraph",
      subgraphNodes: [],
      subgraphEdges: [],
      subgraphInputs: [],
      subgraphOutputs: [],
    },
  },
};

interface UseNodeFactoriesArgs {
  nodes: Node<FlowNodeData>[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<
    React.SetStateAction<import("@xyflow/react").Edge[]>
  >;
  availablePipelineIds: string[];
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  handlePipelineSelect: (nodeId: string, newPipelineId: string | null) => void;
  selectedNodeIds: string[];
  setSelectedNodeIds: (ids: string[]) => void;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
  pendingNodePosition: { x: number; y: number } | null;
  setPendingNodePosition: (pos: { x: number; y: number } | null) => void;
}

export function useNodeFactories({
  nodes,
  setNodes,
  setEdges,
  availablePipelineIds,
  portsMap,
  handlePipelineSelect,
  selectedNodeIds,
  setSelectedNodeIds,
  spoutOutputAvailable,
  ndiOutputAvailable,
  syphonOutputAvailable,
  pendingNodePosition,
  setPendingNodePosition,
}: UseNodeFactoriesArgs) {
  const existingIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);

  /** Generic factory: creates a node from the NODE_DEFAULTS config map. */
  const addNode = useCallback(
    (
      key: NodeTypeKey,
      position?: { x: number; y: number },
      extraData?: Partial<FlowNodeData>
    ) => {
      const def = NODE_DEFAULTS[key];
      const id = generateNodeId(def.idPrefix, existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: def.type,
        position: position ?? {
          x: def.defaultX,
          y: 50 + nodes.length * 100,
        },
        ...(def.style ? { style: def.style } : {}),
        data: {
          label: id,
          ...def.data,
          ...extraData,
        } as FlowNodeData,
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const handleNodeTypeSelect = useCallback(
    (
      type:
        | "source"
        | "pipeline"
        | "sink"
        | "primitive"
        | "control"
        | "math"
        | "note"
        | "output"
        | "slider"
        | "knobs"
        | "xypad"
        | "tuple"
        | "reroute"
        | "image"
        | "vace"
        | "midi"
        | "bool"
        | "subgraph",
      subType?: string
    ) => {
      if (!pendingNodePosition) return;

      // Enforce single source and single sink
      if (type === "source") {
        const hasSource = nodes.some(n => n.data.nodeType === "source");
        if (hasSource) {
          toast.warning("Only one Source node is allowed");
          setPendingNodePosition(null);
          return;
        }
      }
      if (type === "sink") {
        const hasSink = nodes.some(n => n.data.nodeType === "sink");
        if (hasSink) {
          toast.warning("Only one Sink node is allowed");
          setPendingNodePosition(null);
          return;
        }
      }

      if (type === "control") {
        if (subType === "float" || subType === "int" || subType === "string") {
          addNode(`control_${subType}` as NodeTypeKey, pendingNodePosition);
        }
      } else if (type === "pipeline") {
        addNode("pipeline", pendingNodePosition, {
          availablePipelineIds,
          pipelinePortsMap: portsMap,
          onPipelineSelect: handlePipelineSelect,
        });
      } else if (type === "output") {
        const defaultType = spoutOutputAvailable
          ? "spout"
          : ndiOutputAvailable
            ? "ndi"
            : syphonOutputAvailable
              ? "syphon"
              : "spout";
        const defaultNames: Record<string, string> = {
          spout: "ScopeOut",
          ndi: "Scope",
          syphon: "Scope",
        };
        addNode("output", pendingNodePosition, {
          outputSinkType: defaultType,
          outputSinkName: defaultNames[defaultType] || "Scope",
        });
      } else {
        addNode(type as NodeTypeKey, pendingNodePosition);
      }

      setPendingNodePosition(null);
    },
    [
      pendingNodePosition,
      addNode,
      availablePipelineIds,
      portsMap,
      handlePipelineSelect,
      spoutOutputAvailable,
      ndiOutputAvailable,
      syphonOutputAvailable,
      setPendingNodePosition,
    ]
  );

  const handleDeleteNodes = useCallback(
    (nodeIds: string[]) => {
      // Never delete boundary nodes
      const PROTECTED = new Set([
        "__sg_boundary_input__",
        "__sg_boundary_output__",
      ]);
      const idSet = new Set(nodeIds.filter(id => !PROTECTED.has(id)));
      if (idSet.size === 0) return;
      setNodes(nds => nds.filter(n => !idSet.has(n.id)));
      setEdges(eds =>
        eds.filter(e => !idSet.has(e.source) && !idSet.has(e.target))
      );
      setSelectedNodeIds(selectedNodeIds.filter(id => !idSet.has(id)));
    },
    [setNodes, setEdges, selectedNodeIds, setSelectedNodeIds]
  );

  /** Strip non-serializable data from a node for storage inside a subgraph. */
  const stripForSubgraph = (data: FlowNodeData): Record<string, unknown> => {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === "function") continue;
      if (
        typeof value === "object" &&
        value !== null &&
        !Array.isArray(value) &&
        Object.getPrototypeOf(value) !== Object.prototype
      ) {
        continue; // skip MediaStream etc.
      }
      result[key] = value;
    }
    return result;
  };

  /**
   * Create a subgraph from the currently selected nodes.
   * - Detects "dangling" edges (crossing the selection boundary) and
   *   creates SubgraphPort entries for each.
   * - Removes selected nodes, adds a new subgraph node at their centroid.
   * - Reconnects external edges to the subgraph's exposed port handles.
   */
  const createSubgraphFromSelection = useCallback(
    (
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      selectedIds: string[]
    ) => {
      if (selectedIds.length < 1) {
        toast.warning("Select at least one node to create a subgraph");
        return;
      }

      const selSet = new Set(selectedIds);
      const selectedNodes = currentNodes.filter(n => selSet.has(n.id));
      const otherNodes = currentNodes.filter(n => !selSet.has(n.id));

      // Edges fully inside the selection
      const innerEdges = currentEdges.filter(
        e => selSet.has(e.source) && selSet.has(e.target)
      );
      // Edges crossing the boundary
      const incomingEdges = currentEdges.filter(
        e => !selSet.has(e.source) && selSet.has(e.target)
      );
      const outgoingEdges = currentEdges.filter(
        e => selSet.has(e.source) && !selSet.has(e.target)
      );
      // Edges not touching the selection at all
      const externalEdges = currentEdges.filter(
        e => !selSet.has(e.source) && !selSet.has(e.target)
      );

      // Calculate centroid for the subgraph node position
      const centroidX =
        selectedNodes.reduce((sum, n) => sum + n.position.x, 0) /
        selectedNodes.length;
      const centroidY =
        selectedNodes.reduce((sum, n) => sum + n.position.y, 0) /
        selectedNodes.length;

      // Build subgraph input ports from incoming edges
      const subgraphInputs: SubgraphPort[] = [];
      const inputPortNameCounts = new Map<string, number>();
      for (const edge of incomingEdges) {
        const parsed = parseHandleId(edge.targetHandle);
        if (!parsed) continue;
        const baseName = parsed.name;
        const count = inputPortNameCounts.get(baseName) ?? 0;
        inputPortNameCounts.set(baseName, count + 1);
        const portName = count > 0 ? `${baseName}_${count}` : baseName;

        // Determine paramType from the target node's parameter inputs
        let paramType: SubgraphPort["paramType"];
        if (parsed.kind === "param") {
          const targetNode = currentNodes.find(n => n.id === edge.target);
          const pInput = targetNode?.data.parameterInputs?.find(
            p => p.name === parsed.name
          );
          paramType = pInput?.type ?? "number";
        }

        subgraphInputs.push({
          name: portName,
          portType: parsed.kind,
          paramType,
          innerNodeId: edge.target,
          innerHandleId: edge.targetHandle || "",
        });
      }

      // Build subgraph output ports from outgoing edges
      const subgraphOutputs: SubgraphPort[] = [];
      const outputPortNameCounts = new Map<string, number>();
      for (const edge of outgoingEdges) {
        const parsed = parseHandleId(edge.sourceHandle);
        if (!parsed) continue;
        const baseName = parsed.name;
        const count = outputPortNameCounts.get(baseName) ?? 0;
        outputPortNameCounts.set(baseName, count + 1);
        const portName = count > 0 ? `${baseName}_${count}` : baseName;

        // Determine paramType from the source node's parameter outputs
        let paramType: SubgraphPort["paramType"];
        if (parsed.kind === "param") {
          const sourceNode = currentNodes.find(n => n.id === edge.source);
          const pOutput = sourceNode?.data.parameterOutputs?.find(
            p => p.name === parsed.name
          );
          paramType = pOutput?.type ?? "number";
        }

        subgraphOutputs.push({
          name: portName,
          portType: parsed.kind,
          paramType,
          innerNodeId: edge.source,
          innerHandleId: edge.sourceHandle || "",
        });
      }

      // Serialize inner nodes and edges
      const serializedInnerNodes: SerializedSubgraphNode[] = selectedNodes.map(
        n => {
          const w =
            n.width ??
            n.measured?.width ??
            (typeof n.style?.width === "number" ? n.style.width : undefined);
          const h =
            n.height ??
            n.measured?.height ??
            (typeof n.style?.height === "number" ? n.style.height : undefined);
          return {
            id: n.id,
            type: n.data.nodeType || n.type || "pipeline",
            position: {
              x: n.position.x - centroidX + 200,
              y: n.position.y - centroidY + 200,
            },
            ...(w && !Number.isNaN(w) ? { width: w } : {}),
            ...(h && !Number.isNaN(h) ? { height: h } : {}),
            data: stripForSubgraph(n.data),
          };
        }
      );
      const serializedInnerEdges: SerializedSubgraphEdge[] = innerEdges.map(
        e => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle ?? null,
          target: e.target,
          targetHandle: e.targetHandle ?? null,
        })
      );

      // Create the subgraph node
      const sgId = generateNodeId("subgraph", existingIds);
      const subgraphNode: Node<FlowNodeData> = {
        id: sgId,
        type: "subgraph",
        position: { x: centroidX, y: centroidY },
        data: {
          label: "Subgraph",
          nodeType: "subgraph",
          subgraphNodes: serializedInnerNodes,
          subgraphEdges: serializedInnerEdges,
          subgraphInputs,
          subgraphOutputs,
        },
      };

      // Remap incoming edges to point to the subgraph's input ports
      const remappedIncoming: Edge[] = incomingEdges.map((edge, i) => {
        const port = subgraphInputs[i];
        return {
          ...edge,
          id: `e-${sgId}-in-${port.name}`,
          target: sgId,
          targetHandle: buildHandleId(port.portType, port.name),
        };
      });

      // Remap outgoing edges to come from the subgraph's output ports
      const remappedOutgoing: Edge[] = outgoingEdges.map((edge, i) => {
        const port = subgraphOutputs[i];
        return {
          ...edge,
          id: `e-${sgId}-out-${port.name}`,
          source: sgId,
          sourceHandle: buildHandleId(port.portType, port.name),
        };
      });

      const newNodes = [...otherNodes, subgraphNode];
      const newEdges = [
        ...externalEdges,
        ...remappedIncoming,
        ...remappedOutgoing,
      ];

      setNodes(newNodes);
      setEdges(newEdges);
      setSelectedNodeIds([sgId]);
      toast.success(
        `Created subgraph with ${selectedNodes.length} node${selectedNodes.length !== 1 ? "s" : ""}`
      );
    },
    [existingIds, setNodes, setEdges, setSelectedNodeIds]
  );

  /**
   * Unpack a subgraph node – dissolve it back into individual nodes.
   */
  const unpackSubgraph = useCallback(
    (
      nodeId: string,
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[]
    ) => {
      const sgNode = currentNodes.find(n => n.id === nodeId);
      if (!sgNode || sgNode.data.nodeType !== "subgraph") return;

      const innerNodesSerialized = sgNode.data.subgraphNodes ?? [];
      const innerEdgesSerialized = sgNode.data.subgraphEdges ?? [];
      const sgInputs = sgNode.data.subgraphInputs ?? [];
      const sgOutputs = sgNode.data.subgraphOutputs ?? [];

      // Re-position inner nodes relative to the subgraph node's position
      const restoredNodes: Node<FlowNodeData>[] = innerNodesSerialized.map(
        n => {
          const sizeProps =
            n.width != null || n.height != null
              ? {
                  width: n.width ?? undefined,
                  height: n.height ?? undefined,
                  style: {
                    width: n.width ?? undefined,
                    height: n.height ?? undefined,
                  },
                }
              : {};
          return {
            id: n.id,
            type: n.type,
            position: {
              x: n.position.x + sgNode.position.x - 200,
              y: n.position.y + sgNode.position.y - 200,
            },
            ...sizeProps,
            data: {
              ...n.data,
              label: n.data.label ?? n.id,
            } as FlowNodeData,
          };
        }
      );

      // Restore inner edges
      const restoredEdges: Edge[] = innerEdgesSerialized.map(e => ({
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle ?? undefined,
        target: e.target,
        targetHandle: e.targetHandle ?? undefined,
      }));

      // Build port lookup maps: port name → inner target/source
      const inputMap = new Map<string, SubgraphPort>();
      for (const p of sgInputs) inputMap.set(p.name, p);
      const outputMap = new Map<string, SubgraphPort>();
      for (const p of sgOutputs) outputMap.set(p.name, p);

      // Remap edges that were connected to the subgraph
      const externalEdges: Edge[] = [];
      const otherEdges: Edge[] = [];
      for (const edge of currentEdges) {
        if (edge.target === nodeId) {
          // Incoming edge → remap to inner node
          const parsed = parseHandleId(edge.targetHandle);
          const port = parsed ? inputMap.get(parsed.name) : undefined;
          if (port) {
            externalEdges.push({
              ...edge,
              id: `e-unpack-${edge.source}-${port.innerNodeId}`,
              target: port.innerNodeId,
              targetHandle: port.innerHandleId,
            });
          }
        } else if (edge.source === nodeId) {
          // Outgoing edge → remap to inner node
          const parsed = parseHandleId(edge.sourceHandle);
          const port = parsed ? outputMap.get(parsed.name) : undefined;
          if (port) {
            externalEdges.push({
              ...edge,
              id: `e-unpack-${port.innerNodeId}-${edge.target}`,
              source: port.innerNodeId,
              sourceHandle: port.innerHandleId,
            });
          }
        } else {
          otherEdges.push(edge);
        }
      }

      // Remove the subgraph node, add inner nodes
      const remainingNodes = currentNodes.filter(n => n.id !== nodeId);
      setNodes([...remainingNodes, ...restoredNodes]);
      setEdges([...otherEdges, ...restoredEdges, ...externalEdges]);

      const innerIds = restoredNodes.map(n => n.id);
      setSelectedNodeIds(innerIds);
      toast.success(
        `Unpacked subgraph into ${restoredNodes.length} node${restoredNodes.length !== 1 ? "s" : ""}`
      );
    },
    [setNodes, setEdges, setSelectedNodeIds]
  );

  return {
    handleNodeTypeSelect,
    handleDeleteNodes,
    createSubgraphFromSelection,
    unpackSubgraph,
  };
}
