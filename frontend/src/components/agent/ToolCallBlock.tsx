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
      ? "h-3 w-3 animate-spin text-[#8c8c8d] shrink-0"
      : call.status === "error"
        ? "h-3 w-3 text-red-400 shrink-0"
        : "h-3 w-3 text-emerald-400/80 shrink-0";

  const hasDetail = !!(call.input && Object.keys(call.input).length > 0);

  return (
    <div>
      <button
        type="button"
        onClick={() => hasDetail && setExpanded(v => !v)}
        disabled={!hasDetail}
        className="w-full flex items-center gap-1.5 px-1 py-0.5 text-[11px] text-[#8c8c8d] hover:text-[#cfd3da] disabled:hover:text-[#8c8c8d] disabled:cursor-default text-left"
      >
        {hasDetail ? (
          expanded ? (
            <ChevronDown className="h-2.5 w-2.5 shrink-0" />
          ) : (
            <ChevronRight className="h-2.5 w-2.5 shrink-0" />
          )
        ) : (
          <span className="w-2.5 shrink-0" />
        )}
        <Icon className={iconClass} />
        <span className="font-mono shrink-0">{call.name}</span>
        {call.summary && (
          <span className="truncate text-[#6e6e6e]">— {call.summary}</span>
        )}
      </button>
      {expanded && hasDetail && (
        <div className="pl-5 pr-1 pb-1 pt-0.5 text-[10px] text-[#8c8c8d] font-mono space-y-1">
          {call.input && Object.keys(call.input).length > 0 && (
            <pre className="whitespace-pre-wrap break-words bg-[#0f0f0f]/60 rounded px-2 py-1 border border-[rgba(255,255,255,0.04)]">
              {JSON.stringify(call.input, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
