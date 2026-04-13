import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import type { FlowNodeData } from "../../lib/graphUtils";

interface AddNodeModalProps {
  open: boolean;
  onClose: () => void;
  onSelectNodeType: (
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
}

interface NodeCatalogItem {
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
    | "custom_node";
  subType?: string;
  name: string;
  description: string;
  color: string;
  category: string;
  /** Full definition for custom nodes (inputs/outputs/params). */
  customNodeDef?: Record<string, unknown>;
}

const NODE_CATALOG: NodeCatalogItem[] = [
  {
    type: "source",
    name: "Source",
    description: "Input node for the workflow",
    color: "#4ade80",
    category: "I/O",
  },
  {
    type: "pipeline",
    name: "Node",
    description: "Video processing node (pick a model after dropping it)",
    color: "#60a5fa",
    category: "I/O",
  },
  {
    type: "control",
    subType: "float",
    name: "FloatControl",
    description:
      "Animated float output using patterns like sine, bounce, random walk",
    color: "#38bdf8",
    category: "Controls",
  },
  {
    type: "control",
    subType: "int",
    name: "IntControl",
    description:
      "Animated integer output using the same movement patterns as FloatControl",
    color: "#38bdf8",
    category: "Controls",
  },
  {
    type: "midi",
    name: "MIDI",
    description:
      "Receive MIDI CC/Note input and output normalized values to parameters",
    color: "#06b6d4",
    category: "Controls",
  },
  {
    type: "math",
    name: "Math",
    description: "Perform arithmetic operations on two numeric inputs",
    color: "#38bdf8",
    category: "Utility",
  },
  {
    type: "xypad",
    name: "XY Pad",
    description: "2D touch pad outputting X and Y values for two-axis control",
    color: "#38bdf8",
    category: "UI",
  },
  // Purple
  {
    type: "slider",
    name: "Slider",
    description:
      "Horizontal slider for a single numeric value with min/max/step",
    color: "#a78bfa",
    category: "UI",
  },
  // Pink
  {
    type: "knobs",
    name: "Knobs",
    description:
      "Multi-knob console with dynamic add/remove and per-knob range",
    color: "#f472b6",
    category: "UI",
  },
  // Pink
  {
    type: "image",
    name: "Image",
    description: "Pick an image from assets to use as reference or VACE input",
    color: "#f472b6",
    category: "I/O",
  },
  // Red
  {
    type: "output",
    name: "Output",
    description: "Send video to Spout, NDI, or Syphon receivers",
    color: "#f87171",
    category: "I/O",
  },
  // Orange
  {
    type: "sink",
    name: "Sink",
    description: "Output node for the workflow",
    color: "#fb923c",
    category: "I/O",
  },
  // Red
  {
    type: "record",
    name: "Record",
    description: "Record the output stream to MP4",
    color: "#ef4444",
    category: "I/O",
  },
  {
    type: "tuple",
    name: "Tuple",
    description:
      "Dynamic list of numbers with ordering constraints (e.g. denoising steps)",
    color: "#fb923c",
    category: "UI",
  },
  // Yellow
  {
    type: "control",
    subType: "string",
    name: "StringControl",
    description: "Cycles through a list of strings using movement patterns",
    color: "#fbbf24",
    category: "Controls",
  },
  {
    type: "note",
    name: "Note",
    description: "Add a text annotation to the graph",
    color: "#fbbf24",
    category: "Utility",
  },
  // Violet
  {
    type: "vace",
    name: "VACE",
    description:
      "Bundle VACE parameters (context scale, reference images) for pipeline conditioning",
    color: "#a78bfa",
    category: "I/O",
  },
  // Pink
  {
    type: "lora",
    name: "LoRA",
    description:
      "Configure LoRA adapters (file, scale, merge mode) and connect to a pipeline node",
    color: "#f472b6",
    category: "I/O",
  },
  // Gray
  {
    type: "primitive",
    name: "Primitive",
    description: "Adaptive value node — auto-detects type from connected input",
    color: "#9ca3af",
    category: "Values",
  },
  {
    type: "bool",
    name: "Bool",
    description:
      "Convert number to boolean — gate (momentary) or toggle (latching)",
    color: "#34d399",
    category: "Utility",
  },
  {
    type: "trigger",
    name: "Trigger",
    description: "Momentary pulse button — fires a boolean bang on click",
    color: "#f97316",
    category: "Utility",
  },
  {
    type: "reroute",
    name: "Reroute",
    description: "Pass-through dot to organize long connection lines",
    color: "#9ca3af",
    category: "Utility",
  },
  // Cyan – Subgraph
  {
    type: "subgraph",
    name: "Subgraph",
    description: "Container node that groups nodes into a reusable sub-graph",
    color: "#06b6d4",
    category: "Utility",
  },
  {
    type: "tempo",
    name: "Tempo",
    description:
      "Beat clock with BPM, phase, and count outputs from Ableton Link or MIDI Clock",
    color: "#f59e0b",
    category: "Controls",
  },
  {
    type: "prompt_list",
    name: "Prompt Cycle",
    description:
      "Cycle through a list of prompts driven by an external signal (e.g. beat count)",
    color: "#8b5cf6",
    category: "Controls",
  },
  {
    type: "prompt_blend",
    name: "Prompt List",
    description:
      "Blend multiple weighted prompts with linear or SLERP interpolation",
    color: "#fbbf24",
    category: "Controls",
  },
  {
    type: "scheduler",
    name: "Scheduler",
    description:
      "Time-based trigger scheduler that fires named outputs at specific time points",
    color: "#f97316",
    category: "Utility",
  },
];

const CATEGORIES = [
  "All",
  "I/O",
  "Values",
  "Controls",
  "UI",
  "Utility",
  "Plugins",
];

interface TooltipState {
  text: string;
  x: number;
  y: number;
}

function NodeTile({
  item,
  onSelect,
}: {
  item: NodeCatalogItem;
  onSelect: () => void;
}) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const handleMouseEnter = (e: React.MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    timerRef.current = setTimeout(() => {
      setTooltip({
        text: item.description,
        x: rect.left + rect.width / 2,
        y: rect.top - 8,
      });
    }, 800);
  };

  const handleMouseLeave = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setTooltip(null);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <>
      <button
        onClick={onSelect}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="flex items-center gap-1.5 p-2 h-[56px] rounded-lg bg-[#242424] border border-[rgba(119,119,119,0.12)] hover:bg-[#2e2e2e] hover:border-[rgba(119,119,119,0.35)] transition-colors text-left group"
      >
        <div
          className="w-2.5 h-2.5 rounded-full shrink-0 transition-transform group-hover:scale-125"
          style={{ backgroundColor: item.color }}
        />
        <span className="text-[11px] font-medium text-[#e0e0e0] leading-tight truncate">
          {item.name}
        </span>
      </button>

      {tooltip && (
        <div
          className="fixed z-[9999] pointer-events-none"
          style={{
            left: tooltip.x,
            top: tooltip.y,
            transform: "translate(-50%, -100%)",
          }}
        >
          <div className="px-2.5 py-1.5 rounded-lg bg-[#111] border border-[rgba(119,119,119,0.3)] text-[11px] text-[#ccc] max-w-[180px] text-center shadow-lg">
            {tooltip.text}
          </div>
          <div
            className="mx-auto mt-[-1px] w-0 h-0"
            style={{
              borderLeft: "5px solid transparent",
              borderRight: "5px solid transparent",
              borderTop: "5px solid rgba(119,119,119,0.3)",
              width: 0,
            }}
          />
        </div>
      )}
    </>
  );
}

export function AddNodeModal({
  open,
  onClose,
  onSelectNodeType,
}: AddNodeModalProps) {
  const [searchText, setSearchText] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [customNodes, setCustomNodes] = useState<NodeCatalogItem[]>([]);

  useEffect(() => {
    if (!open) return;
    fetch("/api/v1/nodes/definitions")
      .then(r => r.json())
      .then(data => {
        // The unified endpoint returns both pipelines (pipeline_meta != null)
        // and plain custom nodes. Pipelines are still added via the hardcoded
        // "Pipeline" catalog entry (placeholder + dropdown), so we filter
        // them out of the plugin listing here to avoid duplication.
        const items: NodeCatalogItem[] = (data.nodes ?? [])
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .filter((n: any) => n.pipeline_meta == null)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .map((n: any) => ({
            type: "custom_node" as const,
            subType: n.node_type_id,
            name: n.display_name || n.node_type_id,
            description: n.description || "",
            color: "#9ca3af",
            category: "Plugins",
            customNodeDef: n,
          }));
        setCustomNodes(items);
      })
      .catch(() => {
        /* ignore — custom nodes just won't appear */
      });
  }, [open]);

  const fullCatalog = useMemo(
    () => [...NODE_CATALOG, ...customNodes],
    [customNodes]
  );

  const filteredItems = useMemo(() => {
    const lowerSearch = searchText.toLowerCase();
    return fullCatalog.filter(item => {
      const matchesSearch =
        !lowerSearch ||
        item.name.toLowerCase().includes(lowerSearch) ||
        item.description.toLowerCase().includes(lowerSearch);
      const matchesCategory =
        activeCategory === "All" || item.category === activeCategory;
      return matchesSearch && matchesCategory;
    });
  }, [searchText, activeCategory, fullCatalog]);

  const handleSelect = useCallback(
    (item: NodeCatalogItem) => {
      if (item.type === "custom_node" && item.customNodeDef) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const def = item.customNodeDef as any;
        onSelectNodeType("custom_node", item.subType, {
          customNodeTypeId: def.node_type_id,
          customNodeDisplayName: def.display_name || def.node_type_id,
          customNodeCategory: def.category || "",
          customNodeInputs: def.inputs || [],
          customNodeOutputs: def.outputs || [],
          customNodeParamDefs: def.params || [],
          customNodeParams: Object.fromEntries(
            (def.params || [])
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              .filter((p: any) => p.default != null)
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              .map((p: any) => [p.name, p.default])
          ),
        });
      } else {
        onSelectNodeType(item.type, item.subType);
      }
      onClose();
      setSearchText("");
      setActiveCategory("All");
    },
    [onSelectNodeType, onClose]
  );

  const handleClose = () => {
    onClose();
    setSearchText("");
    setActiveCategory("All");
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="!max-w-xl w-full p-0 overflow-hidden bg-[#1a1a1a] border border-[rgba(119,119,119,0.2)] rounded-2xl">
        <DialogHeader className="sr-only">
          <DialogTitle>Add Node</DialogTitle>
          <DialogDescription>
            Select the type of node to add to the workflow
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col h-[420px]">
          {/* Search bar */}
          <div className="px-4 pt-4 pb-3 border-b border-[rgba(119,119,119,0.12)]">
            <div className="flex items-center gap-2 px-3 py-2 bg-[#111] rounded-lg border border-[rgba(119,119,119,0.2)]">
              <svg
                className="w-3.5 h-3.5 text-[#666] shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z"
                />
              </svg>
              <input
                type="text"
                value={searchText}
                onChange={e => {
                  setSearchText(e.target.value);
                }}
                placeholder="Search for anything..."
                className="flex-1 bg-transparent text-xs text-[#fafafa] placeholder:text-[#555] focus:outline-none"
                autoFocus
              />
            </div>
          </div>

          {/* Category tabs */}
          <div className="flex items-center gap-1.5 px-4 py-2 border-b border-[rgba(119,119,119,0.12)]">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  activeCategory === cat
                    ? "bg-[#fafafa] text-[#111]"
                    : "text-[#888] hover:text-[#ccc]"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Grid */}
          <div className="flex-1 overflow-y-auto px-4 py-3 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded-full">
            {filteredItems.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#555] text-xs">
                No nodes found{searchText ? ` for "${searchText}"` : ""}
              </div>
            ) : (
              <div className="grid grid-cols-6 gap-1.5">
                {filteredItems.map(item => (
                  <NodeTile
                    key={`${item.type}-${item.subType || ""}`}
                    item={item}
                    onSelect={() => handleSelect(item)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-[rgba(119,119,119,0.12)]">
            <button
              onClick={handleClose}
              className="px-5 py-2 rounded-full bg-[#2a2a2a] border border-[rgba(119,119,119,0.2)] text-xs font-medium text-[#fafafa] hover:bg-[#333] transition-colors"
            >
              Cancel
            </button>
            <span className="text-[10px] text-[#555]">
              {filteredItems.length} node{filteredItems.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
