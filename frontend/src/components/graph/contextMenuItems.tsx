import {
  Camera,
  Workflow,
  Monitor,
  SlidersHorizontal,
  Trash2,
  Type,
  Hash,
  ToggleLeft,
  Sigma,
  StickyNote,
  Send,
  Gauge,
  CircleDot,
  Grid2x2,
  ListOrdered,
  GitBranch,
  Image,
  Sparkles,
  Lock,
  LockOpen,
  Pin,
  PinOff,
  Music,
  FolderOpen,
  PackageOpen,
  BookOpen,
  Zap,
  Circle,
  Layers,
  Clock,
} from "lucide-react";
import type { Node, Edge } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import type { NodeDefinitionDto } from "../../lib/api";
import type { PipelineInfo } from "../../types";
import type { ContextMenuItem } from "./ContextMenu";
import { buildCustomNodeExtraData } from "./customNodeExtraData";

/* ── Pane (canvas) context menu ──────────────────────────────────────────── */

type NodeTypeSelectFn = (
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
) => void;

export function buildPaneMenuItems(deps: {
  handleNodeTypeSelect: NodeTypeSelectFn;
  selectedNodeIds: string[];
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  createSubgraphFromSelection: (
    nodes: Node<FlowNodeData>[],
    edges: Edge[],
    selectedIds: string[]
  ) => void;
  onOpenBlueprints: () => void;
  /** Installed pipelines, used to surface plugin-provided pipelines via search. */
  installedPipelines?: Record<string, PipelineInfo> | null;
  /** Node definitions from /api/v1/nodes/definitions, used to surface plugin
   *  custom nodes via search. */
  customNodeDefs?: NodeDefinitionDto[];
}): ContextMenuItem[] {
  const {
    handleNodeTypeSelect,
    selectedNodeIds,
    nodes,
    edges,
    createSubgraphFromSelection,
    onOpenBlueprints,
    installedPipelines,
    customNodeDefs,
  } = deps;

  const baseItems: ContextMenuItem[] = [
    {
      label: "Source",
      icon: <Camera />,
      onClick: () => handleNodeTypeSelect("source"),
      keywords: ["input", "camera", "video"],
    },
    {
      label: "Pipeline",
      icon: <Workflow />,
      onClick: () => handleNodeTypeSelect("pipeline"),
      keywords: ["process", "effect", "filter"],
    },
    {
      label: "Sink",
      icon: <Monitor />,
      onClick: () => handleNodeTypeSelect("sink"),
      keywords: ["output", "display", "preview"],
    },
    {
      label: "Output",
      icon: <Send />,
      onClick: () => handleNodeTypeSelect("output"),
      keywords: ["spout", "ndi", "syphon", "send"],
    },
    {
      label: "Record",
      icon: <Circle />,
      onClick: () => handleNodeTypeSelect("record"),
      keywords: ["record", "recording", "mp4", "save", "capture"],
    },
    {
      label: "Controls",
      icon: <SlidersHorizontal />,
      children: [
        {
          label: "FloatControl",
          icon: <Gauge />,
          onClick: () => handleNodeTypeSelect("control", "float"),
          keywords: ["float", "animated", "sine"],
        },
        {
          label: "IntControl",
          icon: <Hash />,
          onClick: () => handleNodeTypeSelect("control", "int"),
          keywords: ["integer", "animated"],
        },
        {
          label: "StringControl",
          icon: <Type />,
          onClick: () => handleNodeTypeSelect("control", "string"),
          keywords: ["text", "cycle", "animated"],
        },
        {
          label: "MIDI",
          icon: <Music />,
          onClick: () => handleNodeTypeSelect("midi"),
          keywords: ["midi", "controller", "cc", "knob", "fader"],
        },
      ],
    },
    {
      label: "UI",
      icon: <CircleDot />,
      children: [
        {
          label: "Slider",
          icon: <SlidersHorizontal />,
          onClick: () => handleNodeTypeSelect("slider"),
          keywords: ["range", "value"],
        },
        {
          label: "Knobs",
          icon: <CircleDot />,
          onClick: () => handleNodeTypeSelect("knobs"),
          keywords: ["dial", "rotary"],
        },
        {
          label: "XY Pad",
          icon: <Grid2x2 />,
          onClick: () => handleNodeTypeSelect("xypad"),
          keywords: ["pad", "2d", "touch"],
        },
        {
          label: "Tuple",
          icon: <ListOrdered />,
          onClick: () => handleNodeTypeSelect("tuple"),
          keywords: ["list", "numbers", "array"],
        },
      ],
    },
    {
      label: "Utility",
      icon: <Sigma />,
      children: [
        {
          label: "Math",
          icon: <Sigma />,
          onClick: () => handleNodeTypeSelect("math"),
          keywords: ["add", "multiply", "arithmetic"],
        },
        {
          label: "Note",
          icon: <StickyNote />,
          onClick: () => handleNodeTypeSelect("note"),
          keywords: ["comment", "annotation", "text"],
        },
        {
          label: "Bool",
          icon: <ToggleLeft />,
          onClick: () => handleNodeTypeSelect("bool"),
          keywords: ["boolean", "gate", "toggle", "switch", "on", "off"],
        },
        {
          label: "Trigger",
          icon: <Zap />,
          onClick: () => handleNodeTypeSelect("trigger"),
          keywords: ["trigger", "pulse", "bang", "fire", "button"],
        },
        {
          label: "Tempo",
          icon: <Zap />,
          onClick: () => handleNodeTypeSelect("tempo"),
          keywords: ["tempo", "bpm", "beat", "clock", "sync", "link", "midi"],
        },
        {
          label: "Scheduler",
          icon: <Clock />,
          onClick: () => handleNodeTypeSelect("scheduler"),
          keywords: [
            "scheduler",
            "timeline",
            "trigger",
            "time",
            "cue",
            "sequence",
          ],
        },
        {
          label: "Prompt Cycle",
          icon: <StickyNote />,
          onClick: () => handleNodeTypeSelect("prompt_list"),
          keywords: ["prompt", "cycle", "text", "rotate"],
        },
        {
          label: "Prompt List",
          icon: <StickyNote />,
          onClick: () => handleNodeTypeSelect("prompt_blend"),
          keywords: ["prompt", "list", "blend", "weight", "mix"],
        },
        {
          label: "Reroute",
          icon: <GitBranch />,
          onClick: () => handleNodeTypeSelect("reroute"),
          keywords: ["passthrough", "wire", "dot"],
        },
      ],
    },
    {
      label: "Media",
      icon: <Image />,
      onClick: () => handleNodeTypeSelect("image"),
      keywords: [
        "media",
        "image",
        "video",
        "picture",
        "photo",
        "reference",
        "film",
      ],
    },
    {
      label: "Audio",
      icon: <Music />,
      onClick: () => handleNodeTypeSelect("audio"),
      keywords: ["audio", "music", "sound", "wav", "mp3", "flac"],
    },
    {
      label: "VACE",
      icon: <Sparkles />,
      onClick: () => handleNodeTypeSelect("vace"),
      keywords: ["vace", "conditioning", "reference", "frame"],
    },
    {
      label: "LoRA",
      icon: <Layers />,
      onClick: () => handleNodeTypeSelect("lora"),
      keywords: ["lora", "adapter", "finetune", "weight"],
    },
    {
      label: "Primitive",
      icon: <ToggleLeft />,
      onClick: () => handleNodeTypeSelect("primitive"),
      keywords: ["value", "string", "number", "boolean"],
    },
    {
      label: "Subgraph",
      icon: <FolderOpen />,
      onClick: () => handleNodeTypeSelect("subgraph"),
      keywords: ["group", "container", "nest", "bundle"],
    },
    {
      label: "Insert Blueprint...",
      icon: <BookOpen />,
      onClick: onOpenBlueprints,
      keywords: ["blueprint", "preset", "template", "library"],
    },
    ...(selectedNodeIds.length > 0
      ? [
          {
            label: `Group ${selectedNodeIds.length} node${selectedNodeIds.length !== 1 ? "s" : ""} into Subgraph`,
            icon: <PackageOpen />,
            onClick: () => {
              createSubgraphFromSelection(nodes, edges, selectedNodeIds);
            },
            keywords: ["create", "subgraph", "group", "selection"],
          },
        ]
      : []),
  ];

  // Search-only ghost entries: invisible in the static menu, but discoverable
  // via the search box. One per installed pipeline, one per plugin-provided
  // custom node. Plugin package names are added to keywords so users can find
  // a node by typing e.g. "gesture" → "scope-gesture-pipeline".
  const ghostItems: ContextMenuItem[] = [];

  if (installedPipelines) {
    for (const [pipelineId, info] of Object.entries(installedPipelines)) {
      const keywords = [
        pipelineId,
        info.pluginName ?? "",
        "pipeline",
        ...(info.about?.split(/\s+/).filter(Boolean) ?? []),
      ].filter(Boolean);
      ghostItems.push({
        label: info.name || pipelineId,
        icon: <Workflow />,
        keywords,
        searchOnly: true,
        onClick: () =>
          handleNodeTypeSelect("pipeline", undefined, {
            pipelineId,
          } as Partial<FlowNodeData>),
      });
    }
  }

  if (customNodeDefs) {
    for (const def of customNodeDefs) {
      // Mirror AddNodeModal filtering: skip pipelines (handled above) and the
      // built-in scheduler (already a top-level entry).
      if (def.pipeline_meta != null) continue;
      if (def.node_type_id === "scheduler") continue;
      // Surface plugin-provided nodes only — built-ins already have static
      // entries in this menu.
      if (!def.plugin_name) continue;
      const keywords = [
        def.node_type_id,
        def.plugin_name,
        def.category,
        ...(def.description?.split(/\s+/).filter(Boolean) ?? []),
      ].filter(Boolean);
      ghostItems.push({
        label: def.display_name || def.node_type_id,
        icon: <PackageOpen />,
        keywords,
        searchOnly: true,
        onClick: () =>
          handleNodeTypeSelect(
            "custom_node",
            def.node_type_id,
            buildCustomNodeExtraData(def)
          ),
      });
    }
  }

  return [...baseItems, ...ghostItems];
}

/* ── Node context menu ───────────────────────────────────────────────────── */

export function buildNodeMenuItems(deps: {
  contextNodeId: string;
  selectedNodeIds: string[];
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  handleDeleteNodes: (ids: string[]) => void;
  handleEnterSubgraph: (nodeId: string) => void;
  unpackSubgraph: (
    nodeId: string,
    nodes: Node<FlowNodeData>[],
    edges: Edge[]
  ) => void;
  createSubgraphFromSelection: (
    nodes: Node<FlowNodeData>[],
    edges: Edge[],
    selectedIds: string[]
  ) => void;
}): ContextMenuItem[] {
  const {
    contextNodeId,
    selectedNodeIds,
    nodes,
    edges,
    setNodes,
    handleDeleteNodes,
    handleEnterSubgraph,
    unpackSubgraph,
    createSubgraphFromSelection,
  } = deps;

  const isInSelection = selectedNodeIds.includes(contextNodeId);
  const targetIds =
    isInSelection && selectedNodeIds.length > 1
      ? selectedNodeIds
      : [contextNodeId];
  const count = targetIds.length;
  const targetNodes = nodes.filter(n => targetIds.includes(n.id));
  const allLocked = targetNodes.every(n => !!n.data.locked);
  const allPinned = targetNodes.every(n => !!n.data.pinned);
  const isSingleSubgraph =
    count === 1 && targetNodes[0]?.data.nodeType === "subgraph";
  const canCreateSubgraph =
    count >= 1 && !targetNodes.every(n => n.data.nodeType === "subgraph");

  return [
    ...(isSingleSubgraph
      ? [
          {
            label: "Enter Subgraph",
            icon: <FolderOpen />,
            onClick: () => handleEnterSubgraph(targetIds[0]),
          },
          {
            label: "Unpack Subgraph",
            icon: <PackageOpen />,
            onClick: () => unpackSubgraph(targetIds[0], nodes, edges),
          },
        ]
      : []),
    ...(canCreateSubgraph
      ? [
          {
            label: "Group into Subgraph",
            icon: <PackageOpen />,
            onClick: () => createSubgraphFromSelection(nodes, edges, targetIds),
          },
        ]
      : []),
    {
      label: allLocked
        ? count > 1
          ? `Unlock ${count} nodes`
          : "Unlock"
        : count > 1
          ? `Lock ${count} nodes`
          : "Lock",
      icon: allLocked ? <LockOpen /> : <Lock />,
      onClick: () => {
        const newLocked = !allLocked;
        setNodes(nds =>
          nds.map(n =>
            targetIds.includes(n.id)
              ? { ...n, data: { ...n.data, locked: newLocked } }
              : n
          )
        );
      },
    },
    {
      label: allPinned
        ? count > 1
          ? `Unpin ${count} nodes`
          : "Unpin"
        : count > 1
          ? `Pin ${count} nodes`
          : "Pin",
      icon: allPinned ? <PinOff /> : <Pin />,
      onClick: () => {
        const newPinned = !allPinned;
        setNodes(nds =>
          nds.map(n =>
            targetIds.includes(n.id)
              ? {
                  ...n,
                  draggable: !newPinned,
                  data: { ...n.data, pinned: newPinned },
                }
              : n
          )
        );
      },
    },
    {
      label: count > 1 ? `Delete ${count} nodes` : "Delete",
      icon: <Trash2 />,
      onClick: () => handleDeleteNodes(targetIds),
      danger: true,
    },
  ];
}
