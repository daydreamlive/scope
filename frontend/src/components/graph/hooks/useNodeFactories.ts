import { useCallback, useMemo } from "react";
import type { Node } from "@xyflow/react";
import { generateNodeId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";

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
  | "vace";

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
    idPrefix: "image",
    defaultX: 50,
    style: { width: 160, height: 140 },
    data: {
      label: "Image",
      nodeType: "image",
      imagePath: "",
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
      parameterOutputs: [{ name: "__vace", type: "string", defaultValue: "" }],
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
        | "vace",
      subType?: string
    ) => {
      if (!pendingNodePosition) return;

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
      const idSet = new Set(nodeIds);
      setNodes(nds => nds.filter(n => !idSet.has(n.id)));
      setEdges(eds =>
        eds.filter(e => !idSet.has(e.source) && !idSet.has(e.target))
      );
      setSelectedNodeIds(selectedNodeIds.filter(id => !idSet.has(id)));
    },
    [setNodes, setEdges, selectedNodeIds, setSelectedNodeIds]
  );

  return {
    handleNodeTypeSelect,
    handleDeleteNodes,
  };
}
