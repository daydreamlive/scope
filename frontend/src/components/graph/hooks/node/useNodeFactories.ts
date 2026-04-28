import { useCallback, useMemo } from "react";
import type { Node } from "@xyflow/react";
import { generateNodeId } from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import {
  deserializeNodes,
  deserializeEdges,
} from "../../utils/subgraphSerialization";
import { buildEdgeStyle } from "../../constants";
import type { Blueprint } from "../../../../data/blueprints/types";
import { resetAutoHeightNodes } from "../../utils/nodeEnrichment";

import type { EnrichNodesDeps } from "../graph/useGraphPersistence";
import { enrichNodes } from "../graph/useGraphPersistence";

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
  | "audio"
  | "vace"
  | "lora"
  | "midi"
  | "bool"
  | "trigger"
  | "subgraph"
  | "subgraph_input"
  | "subgraph_output"
  | "record"
  | "tempo"
  | "prompt_list"
  | "prompt_blend"
  | "scheduler"
  | "custom_node";

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
  audio: {
    type: "audio",
    idPrefix: "audio",
    defaultX: 50,
    style: { width: 160, height: 100 },
    data: {
      label: "Audio",
      nodeType: "audio",
      audioPath: "",
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
  lora: {
    type: "lora",
    idPrefix: "lora",
    defaultX: 50,
    style: { width: 220 },
    data: {
      label: "LoRA",
      nodeType: "lora",
      loras: [],
      loraMergeMode: "permanent_merge",
      parameterOutputs: [{ name: "__loras", type: "string", defaultValue: "" }],
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
  trigger: {
    type: "trigger",
    idPrefix: "trigger",
    defaultX: 50,
    data: {
      label: "Trigger",
      nodeType: "trigger",
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
  record: {
    type: "record",
    idPrefix: "record",
    defaultX: 900,
    style: { width: 180, height: 100 },
    data: {
      label: "Record",
      nodeType: "record",
      parameterInputs: [
        { name: "trigger", type: "boolean", defaultValue: false },
      ],
    },
  },
  tempo: {
    type: "tempo",
    idPrefix: "tempo",
    defaultX: 50,
    data: {
      label: "Tempo",
      nodeType: "tempo",
      tempoBpm: null,
      tempoBeatPhase: 0,
      tempoBeatCount: 0,
      tempoBarPosition: 0,
      tempoEnabled: false,
      tempoSourceType: null,
      tempoBeatsPerBar: 4,
      tempoQuantizeMode: "none",
      tempoLookaheadMs: 0,
      tempoBeatResetRate: "none",
      parameterOutputs: [
        { name: "bpm", type: "number", defaultValue: 0 },
        { name: "beat_phase", type: "number", defaultValue: 0 },
        { name: "beat_count", type: "number", defaultValue: 0 },
        { name: "bar_position", type: "number", defaultValue: 0 },
        { name: "is_playing", type: "number", defaultValue: 0 },
      ],
    },
  },
  prompt_list: {
    type: "prompt_list",
    idPrefix: "prompt_list",
    defaultX: 50,
    data: {
      label: "Prompt Cycle",
      nodeType: "prompt_list",
      promptListItems: ["prompt 1", "prompt 2"],
      promptListActiveIndex: 0,
      promptListActiveText: "prompt 1",
      promptListCycleValue: -1,
      parameterInputs: [{ name: "cycle", type: "number", defaultValue: 0 }],
      parameterOutputs: [{ name: "prompt", type: "string", defaultValue: "" }],
    },
  },
  prompt_blend: {
    type: "prompt_blend",
    idPrefix: "prompt_blend",
    defaultX: 50,
    data: {
      label: "Prompt List",
      nodeType: "prompt_blend",
      promptBlendItems: [
        { text: "prompt 1", weight: 50 },
        { text: "prompt 2", weight: 50 },
      ],
      promptBlendMethod: "linear",
      parameterInputs: [
        { name: "prompt_0", type: "string", defaultValue: "" },
        { name: "prompt_1", type: "string", defaultValue: "" },
      ],
      parameterOutputs: [{ name: "prompts", type: "string", defaultValue: "" }],
    },
  },
  scheduler: {
    type: "scheduler",
    idPrefix: "scheduler",
    defaultX: 50,
    data: {
      label: "Scheduler",
      nodeType: "scheduler",
      schedulerTriggers: [],
      schedulerDuration: 30,
      schedulerLoop: false,
      schedulerElapsed: 0,
      schedulerIsPlaying: false,
      schedulerFireCounts: {},
      schedulerTickCount: 0,
      parameterOutputs: [
        { name: "elapsed", type: "number", defaultValue: 0 },
        { name: "is_playing", type: "number", defaultValue: 0 },
        { name: "tick", type: "number", defaultValue: 0 },
      ],
    },
  },
  custom_node: {
    type: "custom_node",
    idPrefix: "custom",
    defaultX: 300,
    data: {
      label: "Custom Node",
      nodeType: "custom_node" as const,
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
  setSelectedNodeIds: React.Dispatch<React.SetStateAction<string[]>>;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
  pendingNodePosition: { x: number; y: number } | null;
  setPendingNodePosition: (pos: { x: number; y: number } | null) => void;
  handleEdgeDelete: (edgeId: string) => void;
  enrichDepsRef: React.RefObject<EnrichNodesDeps>;
}

export function useNodeFactories({
  nodes,
  setNodes,
  setEdges,
  availablePipelineIds,
  portsMap,
  handlePipelineSelect,
  setSelectedNodeIds,
  spoutOutputAvailable,
  ndiOutputAvailable,
  syphonOutputAvailable,
  pendingNodePosition,
  setPendingNodePosition,
  handleEdgeDelete,
  enrichDepsRef,
}: UseNodeFactoriesArgs) {
  const existingIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);

  /** Generic factory: creates a node from the NODE_DEFAULTS config map.
   *  Returns the new node's id so callers can wire follow-up state (e.g.
   *  preselecting a pipeline immediately after creation). */
  const addNode = useCallback(
    (
      key: NodeTypeKey,
      position?: { x: number; y: number },
      extraData?: Partial<FlowNodeData>
    ): string => {
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
      setNodes(nds => enrichNodes([...nds, newNode], enrichDepsRef.current));
      return id;
    },
    [existingIds, nodes.length, setNodes, enrichDepsRef]
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
        | "audio"
        | "vace"
        | "lora"
        | "midi"
        | "bool"
        | "trigger"
        | "subgraph"
        | "record"
        | "tempo"
        | "prompt_list"
        | "prompt_blend"
        | "scheduler"
        | "custom_node",
      subType?: string,
      extraData?: Partial<FlowNodeData>
    ) => {
      if (!pendingNodePosition) return;

      if (type === "control") {
        if (subType === "float" || subType === "int" || subType === "string") {
          addNode(`control_${subType}` as NodeTypeKey, pendingNodePosition);
        }
      } else if (type === "pipeline") {
        const preselectedPipelineId = (
          extraData as { pipelineId?: string } | undefined
        )?.pipelineId;
        const newId = addNode("pipeline", pendingNodePosition, {
          availablePipelineIds,
          pipelinePortsMap: portsMap,
          onPipelineSelect: handlePipelineSelect,
        });
        if (preselectedPipelineId) {
          // Mirror the manual dropdown flow so ports/params/streamInputs
          // hydrate identically (see PipelineNode.tsx onPipelineSelect call).
          handlePipelineSelect(newId, preselectedPipelineId);
        }
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
      } else if (type === "custom_node") {
        addNode("custom_node", pendingNodePosition, {
          customNodeTypeId: subType,
          label: subType || "Custom Node",
          ...extraData,
        });
      } else {
        addNode(type as NodeTypeKey, pendingNodePosition);
      }

      setPendingNodePosition(null);
    },
    [
      nodes,
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
      // Use functional updater to avoid stale selectedNodeIds closure
      setSelectedNodeIds(prev => prev.filter(id => !idSet.has(id)));
    },
    [setNodes, setEdges, setSelectedNodeIds]
  );

  const insertBlueprint = useCallback(
    (blueprint: Blueprint, insertPos?: { x: number; y: number }) => {
      const rawNodes = deserializeNodes(blueprint.nodes);
      const rawEdges = deserializeEdges(blueprint.edges);

      if (rawNodes.length === 0) return;

      const currentIds = new Set(nodes.map(n => n.id));
      const idMap = new Map<string, string>();

      for (const node of rawNodes) {
        const prefix = (node.data.nodeType as string) || node.type || "node";
        const newId = generateNodeId(prefix, currentIds);
        idMap.set(node.id, newId);
        currentIds.add(newId);
      }

      // Compute bounding box origin to offset nodes relative to insert position
      const minX = Math.min(...rawNodes.map(n => n.position.x));
      const minY = Math.min(...rawNodes.map(n => n.position.y));
      const targetPos = insertPos ?? { x: 200, y: 200 };

      const sizedNodes = resetAutoHeightNodes(rawNodes);
      const newNodes: Node<FlowNodeData>[] = sizedNodes.map(node => ({
        ...node,
        id: idMap.get(node.id)!,
        position: {
          x: node.position.x - minX + targetPos.x,
          y: node.position.y - minY + targetPos.y,
        },
        selected: true,
      }));

      // Build a lookup map of newId → newNode so we can compute edge styles
      const newNodeMap = new Map(newNodes.map(n => [n.id, n]));

      const newEdges = rawEdges.map((edge, idx) => {
        const newSource = idMap.get(edge.source) ?? edge.source;
        const sourceNode = newNodeMap.get(newSource);
        const style = buildEdgeStyle(sourceNode, edge.sourceHandle);
        return {
          ...edge,
          id: `blueprint_${Date.now()}_${idx}`,
          source: newSource,
          target: idMap.get(edge.target) ?? edge.target,
          type: "default" as const,
          reconnectable: "target" as const,
          animated: false,
          style,
          data: { onDelete: handleEdgeDelete },
        };
      });

      setNodes(nds =>
        enrichNodes(
          [
            ...nds.map(n => (n.selected ? { ...n, selected: false } : n)),
            ...newNodes,
          ],
          enrichDepsRef.current
        )
      );

      if (newEdges.length > 0) {
        setEdges(eds => [...eds, ...newEdges]);
      }
    },
    [nodes, setNodes, setEdges, handleEdgeDelete, enrichDepsRef]
  );

  return {
    handleNodeTypeSelect,
    handleDeleteNodes,
    insertBlueprint,
  };
}
