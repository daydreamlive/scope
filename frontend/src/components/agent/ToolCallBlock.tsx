import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import type { AgentToolCall } from "@/contexts/AgentContext";

export function ToolCallBlock({ call }: { call: AgentToolCall }) {
  const [expanded, setExpanded] = useState(false);

  const Icon =
    call.status === "running"
      ? Loader2
      : call.status === "error"
        ? XCircle
        : CheckCircle2;
  const iconClass =
    call.status === "running"
      ? "h-3.5 w-3.5 animate-spin text-[#b0b0b0]"
      : call.status === "error"
        ? "h-3.5 w-3.5 text-red-400"
        : "h-3.5 w-3.5 text-emerald-400";

  return (
    <div className="rounded-md border border-[rgba(255,255,255,0.06)] bg-[#141414]">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-xs text-[#b0b0b0] hover:text-[#fafafa]"
      >
        {expanded ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <Icon className={iconClass} />
        <span className="font-mono">{call.name}</span>
        {call.summary && (
          <span className="truncate text-[#8c8c8d]">— {call.summary}</span>
        )}
      </button>
      {expanded && (
        <div className="px-3 pb-2 pt-0.5 text-[11px] text-[#8c8c8d] font-mono">
          {call.input && Object.keys(call.input).length > 0 && (
            <pre className="whitespace-pre-wrap break-words bg-[#0f0f0f] rounded px-2 py-1.5 border border-[rgba(255,255,255,0.04)]">
              {JSON.stringify(call.input, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
