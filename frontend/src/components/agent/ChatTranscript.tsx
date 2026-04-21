import { useEffect, useRef } from "react";
import type { AgentMessage, AgentProposal } from "@/contexts/AgentContext";
import { MessageBubble } from "./MessageBubble";
import { WorkflowProposalCard } from "./WorkflowProposalCard";

interface ChatTranscriptProps {
  messages: AgentMessage[];
  pendingProposal: AgentProposal | null;
  onDecide: (approved: boolean, reason?: string) => Promise<void>;
}

export function ChatTranscript({
  messages,
  pendingProposal,
  onDecide,
}: ChatTranscriptProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickyBottomRef = useRef<boolean>(true);

  // Track whether user is at the bottom. If so, auto-scroll; otherwise leave
  // their scroll position alone.
  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    stickyBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  useEffect(() => {
    if (!stickyBottomRef.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, pendingProposal]);

  if (messages.length === 0 && !pendingProposal) {
    return (
      <div className="flex-1 overflow-y-auto p-6 text-sm text-[#8c8c8d]">
        <p className="mb-2 text-[#b0b0b0] font-medium">
          Tell me what you want to build.
        </p>
        <p className="mb-4 leading-relaxed">
          I can pick pipelines, compose workflows, and tune parameters by
          watching the output.
        </p>
        <ul className="space-y-1.5 text-xs">
          <li className="leading-relaxed">
            • "Hyperrealistic scene with 3–5 switchable prompts"
          </li>
          <li className="leading-relaxed">
            • "It's not recognizing depth well"
          </li>
          <li className="leading-relaxed">
            • "Help me record what I'm seeing"
          </li>
        </ul>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      className="flex-1 overflow-y-auto p-4 space-y-3"
    >
      {messages.map(m => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {pendingProposal && !pendingProposal.decision && (
        <WorkflowProposalCard proposal={pendingProposal} onDecide={onDecide} />
      )}
    </div>
  );
}
