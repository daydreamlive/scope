import { useCallback, useMemo } from "react";
import type { Node } from "@xyflow/react";
import { generateNodeId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";

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

  const addSourceNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("input", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "source",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        style: { width: 240, height: 200 },
        data: { label: id, nodeType: "source" },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addPipelineNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("pipeline", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "pipeline",
        position: position ?? { x: 350, y: 50 + nodes.length * 100 },
        data: {
          label: id,
          pipelineId: null,
          nodeType: "pipeline",
          availablePipelineIds,
          pipelinePortsMap: portsMap,
          onPipelineSelect: handlePipelineSelect,
          streamInputs: ["video"],
          streamOutputs: ["video"],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [
      existingIds,
      nodes.length,
      setNodes,
      availablePipelineIds,
      portsMap,
      handlePipelineSelect,
    ]
  );

  const addSinkNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("output", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "sink",
        position: position ?? { x: 650, y: 50 + nodes.length * 100 },
        style: { width: 240, height: 200 },
        data: { label: id, nodeType: "sink" },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addPrimitiveNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("primitive", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "primitive",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Primitive",
          nodeType: "primitive",
          valueType: "string",
          value: "",
          parameterOutputs: [
            {
              name: "value",
              type: "string",
              defaultValue: "",
            },
          ],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addRerouteNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("reroute", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "reroute",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Reroute",
          nodeType: "reroute",
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addControlNode = useCallback(
    (
      controlType: "float" | "int" | "string",
      position?: { x: number; y: number }
    ) => {
      const id = generateNodeId(
        controlType === "float"
          ? "floatControl"
          : controlType === "int"
            ? "intControl"
            : "stringControl",
        existingIds
      );
      const newNode: Node<FlowNodeData> = {
        id,
        type: "control",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label:
            controlType === "float"
              ? "FloatControl"
              : controlType === "int"
                ? "IntControl"
                : "StringControl",
          nodeType: "control",
          controlType,
          controlPattern: "sine",
          controlSpeed: 1.0,
          controlMin: 0,
          controlMax: 1.0,
          controlItems:
            controlType === "string" ? ["item1", "item2", "item3"] : undefined,
          isPlaying: false,
          parameterOutputs: [
            {
              name: "value",
              type: controlType === "string" ? "string" : "number",
              defaultValue: controlType === "string" ? "" : 0,
            },
          ],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addMathNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("math", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "math",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Math",
          nodeType: "math",
          mathOp: "add",
          currentValue: undefined,
          parameterOutputs: [
            {
              name: "value",
              type: "number",
              defaultValue: 0,
            },
          ],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addNoteNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("note", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "note",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Note",
          nodeType: "note",
          noteText: "",
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addSliderNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("slider", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "slider",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Slider",
          nodeType: "slider",
          sliderMin: 0,
          sliderMax: 1,
          sliderStep: 0.01,
          value: 0.5,
          parameterOutputs: [
            { name: "value", type: "number", defaultValue: 0.5 },
          ],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addKnobsNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("knobs", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "knobs",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
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
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addXYPadNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("xypad", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "xypad",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
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
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addTupleNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("tuple", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "tuple",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
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
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addOutputNode = useCallback(
    (position?: { x: number; y: number }) => {
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
      const id = generateNodeId("output_sink", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "output",
        position: position ?? { x: 900, y: 50 + nodes.length * 100 },
        data: {
          label: "Output",
          nodeType: "output",
          outputSinkType: defaultType,
          outputSinkEnabled: false,
          outputSinkName: defaultNames[defaultType] || "Scope",
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [
      existingIds,
      nodes.length,
      setNodes,
      spoutOutputAvailable,
      ndiOutputAvailable,
      syphonOutputAvailable,
    ]
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
        | "reroute",
      subType?: string
    ) => {
      if (!pendingNodePosition) return;

      switch (type) {
        case "source":
          addSourceNode(pendingNodePosition);
          break;
        case "pipeline":
          addPipelineNode(pendingNodePosition);
          break;
        case "sink":
          addSinkNode(pendingNodePosition);
          break;
        case "primitive":
          addPrimitiveNode(pendingNodePosition);
          break;
        case "control":
          if (
            subType === "float" ||
            subType === "int" ||
            subType === "string"
          ) {
            addControlNode(subType, pendingNodePosition);
          }
          break;
        case "math":
          addMathNode(pendingNodePosition);
          break;
        case "note":
          addNoteNode(pendingNodePosition);
          break;
        case "output":
          addOutputNode(pendingNodePosition);
          break;
        case "slider":
          addSliderNode(pendingNodePosition);
          break;
        case "knobs":
          addKnobsNode(pendingNodePosition);
          break;
        case "xypad":
          addXYPadNode(pendingNodePosition);
          break;
        case "tuple":
          addTupleNode(pendingNodePosition);
          break;
        case "reroute":
          addRerouteNode(pendingNodePosition);
          break;
      }

      setPendingNodePosition(null);
    },
    [
      pendingNodePosition,
      addSourceNode,
      addPipelineNode,
      addSinkNode,
      addPrimitiveNode,
      addRerouteNode,
      addControlNode,
      addMathNode,
      addNoteNode,
      addOutputNode,
      addSliderNode,
      addKnobsNode,
      addXYPadNode,
      addTupleNode,
      setPendingNodePosition,
    ]
  );

  const handleDeleteNodes = useCallback(
    (nodeIds: string[]) => {
      const idSet = new Set(nodeIds);
      setNodes(nds => nds.filter(n => !idSet.has(n.id)));
      setEdges(eds => eds.filter(e => !idSet.has(e.source) && !idSet.has(e.target)));
      setSelectedNodeIds(selectedNodeIds.filter(id => !idSet.has(id)));
    },
    [setNodes, setEdges, selectedNodeIds, setSelectedNodeIds]
  );

  return {
    handleNodeTypeSelect,
    handleDeleteNodes,
  };
}
