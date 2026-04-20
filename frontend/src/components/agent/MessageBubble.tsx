import { Loader2 } from "lucide-react";
import type { AgentMessage } from "@/contexts/AgentContext";
import { ToolCallBlock } from "./ToolCallBlock";

export function MessageBubble({ message }: { message: AgentMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="rounded-lg px-3 py-2 text-sm bg-[#1f6feb] text-white max-w-[85%] whitespace-pre-wrap break-words">
          {message.text}
        </div>
      </div>
    );
  }

  const hasContent = message.text.length > 0 || message.toolCalls.length > 0;

  return (
    <div className="flex flex-col gap-1.5 max-w-[92%]">
      {message.toolCalls.length > 0 && (
        <div className="rounded-md border border-[rgba(255,255,255,0.05)] bg-[#121212]/50 px-1.5 py-1 flex flex-col">
          {message.toolCalls.map(tc => (
            <ToolCallBlock key={tc.id} call={tc} />
          ))}
        </div>
      )}
      {message.text && (
        <div className="rounded-lg px-3 py-2 text-sm bg-[#1a1a1a] text-[#e6e6e6] border border-[rgba(255,255,255,0.06)] whitespace-pre-wrap break-words">
          {message.text}
        </div>
      )}
      {message.pending && !hasContent && (
        <div className="inline-flex items-center gap-2 text-xs text-[#8c8c8d] px-2 py-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          Thinking…
        </div>
      )}
    </div>
  );
}
