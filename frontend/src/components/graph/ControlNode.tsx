import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { buildHandleId } from "../../lib/graphUtils";
import {
  NodeCard,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
} from "./node-ui";

type ControlNodeType = Node<FlowNodeData, "control">;

const PARAM_TYPE_COLORS: Record<string, string> = {
  number: "#38bdf8", // sky-400 (for float and int)
  string: "#fbbf24", // amber-400
};

function getControlOutputType(controlType: "float" | "int" | "string"): "number" | "string" {
  return controlType === "string" ? "string" : "number";
}

function getControlTypeColor(controlType: "float" | "int" | "string"): string {
  const outputType = getControlOutputType(controlType);
  return PARAM_TYPE_COLORS[outputType] || "#9ca3af";
}

function getControlTitle(type: "float" | "int" | "string"): string {
  if (type === "float") return "FloatControl";
  if (type === "int") return "IntControl";
  return "StringControl";
}

const PATTERN_OPTIONS = [
  { value: "sine", label: "Sine" },
  { value: "bounce", label: "Bounce" },
  { value: "random_walk", label: "Random Walk" },
  { value: "linear", label: "Linear" },
  { value: "step", label: "Step" },
];

function computePatternValue(
  pattern: "sine" | "bounce" | "random_walk" | "linear" | "step",
  t: number,
  speed: number,
  min: number,
  max: number,
  lastValue: number
): number {
  const range = max - min;
  const phase = (t * speed) % 1;

  switch (pattern) {
    case "sine":
      return min + range * (0.5 + 0.5 * Math.sin(phase * 2 * Math.PI));
    case "bounce": {
      const triangle = phase < 0.5 ? phase * 2 : 2 - phase * 2;
      return min + range * triangle;
    }
    case "random_walk": {
      const step = (Math.random() - 0.5) * 0.1 * range;
      const newValue = lastValue + step;
      return Math.max(min, Math.min(max, newValue));
    }
    case "linear":
      return min + range * phase;
    case "step": {
      const steps = 10;
      const stepIndex = Math.floor(phase * steps);
      return min + (range * stepIndex) / (steps - 1);
    }
    default:
      return min;
  }
}

export function ControlNode({
  id,
  data,
  selected,
}: NodeProps<ControlNodeType>) {
  const { setNodes } = useReactFlow();
  const controlType = data.controlType || "float";
  const pattern = data.controlPattern || "sine";
  const speed = data.controlSpeed ?? 1.0;
  const min = data.controlMin ?? 0;
  const max = data.controlMax ?? 1.0;
  const items = data.controlItems || ["item1", "item2", "item3"];
  const isPlaying = data.isPlaying ?? false;

  const [currentValue, setCurrentValue] = useState<number | string>(
    controlType === "string" ? items[0] || "" : min
  );
  const lastValueRef = useRef<number>(min);
  const startTimeRef = useRef<number>(Date.now());
  const animationFrameRef = useRef<number | undefined>(undefined);

  const color = getControlTypeColor(controlType);
  const dotColorClass = "bg-purple-400";
  const title = getControlTitle(controlType);

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    const animate = () => {
      const now = Date.now();
      const elapsed = (now - startTimeRef.current) / 1000;

      if (controlType === "string") {
        const patternValue = computePatternValue(
          pattern,
          elapsed,
          speed,
          0,
          items.length - 1,
          lastValueRef.current
        );
        lastValueRef.current = patternValue;
        const index = Math.floor(patternValue);
        const clampedIndex = Math.max(0, Math.min(items.length - 1, index));
        setCurrentValue(items[clampedIndex] || "");
      } else {
        const floatValue = computePatternValue(
          pattern,
          elapsed,
          speed,
          min,
          max,
          lastValueRef.current
        );
        lastValueRef.current = floatValue;
        const finalValue = controlType === "int" ? Math.round(floatValue) : floatValue;
        setCurrentValue(finalValue);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, pattern, speed, min, max, controlType, items]);

  const handleTogglePlay = () => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const newIsPlaying = !isPlaying;
        if (newIsPlaying) {
          startTimeRef.current = Date.now();
          lastValueRef.current = typeof currentValue === "number" ? currentValue : min;
        }
        return {
          ...n,
          data: {
            ...n.data,
            isPlaying: newIsPlaying,
          },
        };
      })
    );
  };

  const handlePatternChange = (newPattern: string) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            controlPattern: newPattern as typeof pattern,
          },
        };
      })
    );
  };

  const handleMinChange = (val: string | number) => {
    const numVal = Number(val);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            controlMin: numVal,
          },
        };
      })
    );
  };

  const handleMaxChange = (val: string | number) => {
    const numVal = Number(val);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            controlMax: numVal,
          },
        };
      })
    );
  };

  const handleSpeedChange = (val: string | number) => {
    const numVal = Number(val);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            controlSpeed: numVal,
          },
        };
      })
    );
  };

  const handleItemsChange = (val: string | number) => {
    const itemsStr = String(val);
    const itemsArray = itemsStr
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            controlItems: itemsArray.length > 0 ? itemsArray : ["item1"],
          },
        };
      })
    );
  };

  const itemsDisplay = items.join(", ");

  return (
    <NodeCard selected={selected}>
      <div className={`${NODE_TOKENS.header} justify-between`}>
        <div className="flex items-center gap-2">
          <div className={`w-[10px] h-[10px] rounded-full ${dotColorClass} shrink-0`} />
          <p className={NODE_TOKENS.headerText}>{title}</p>
        </div>
        <button
          onClick={handleTogglePlay}
          className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors"
          type="button"
        >
          {isPlaying ? "⏸" : "▶"}
        </button>
      </div>
      <NodeBody withGap>
        <NodeParamRow label="Pattern">
          <NodePillSelect
            value={pattern}
            onChange={handlePatternChange}
            options={PATTERN_OPTIONS}
          />
        </NodeParamRow>

        {controlType === "string" ? (
          <>
            <NodeParamRow label="Items">
              <NodePillInput
                type="text"
                value={itemsDisplay}
                onChange={handleItemsChange}
              />
            </NodeParamRow>
          </>
        ) : (
          <>
            <NodeParamRow label="Min">
              <NodePillInput
                type="number"
                value={min}
                onChange={handleMinChange}
              />
            </NodeParamRow>
            <NodeParamRow label="Max">
              <NodePillInput
                type="number"
                value={max}
                onChange={handleMaxChange}
              />
            </NodeParamRow>
          </>
        )}

        <NodeParamRow label="Speed">
          <NodePillInput
            type="number"
            value={speed}
            onChange={handleSpeedChange}
            min={0.1}
          />
        </NodeParamRow>

        <NodeParamRow label="Value">
          <NodePill className="opacity-75">
            {typeof currentValue === "number"
              ? controlType === "int"
                ? Math.round(currentValue)
                : currentValue.toFixed(3)
              : currentValue}
          </NodePill>
        </NodeParamRow>
      </NodeBody>
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: 44,
          right: 8,
          backgroundColor: color,
        }}
      />
    </NodeCard>
  );
}
