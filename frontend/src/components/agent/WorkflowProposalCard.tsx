import { useState } from "react";
import { Sparkles, Check, X } from "lucide-react";
import type { AgentProposal } from "@/contexts/AgentContext";
import { Button } from "@/components/ui/button";

interface WorkflowProposalCardProps {
  proposal: AgentProposal;
  onDecide: (approved: boolean, reason?: string) => Promise<void>;
}

export function WorkflowProposalCard({
  proposal,
  onDecide,
}: WorkflowProposalCardProps) {
  const [submitting, setSubmitting] = useState<"approve" | "reject" | null>(
    null
  );
  const [showRejectInput, setShowRejectInput] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [showGraph, setShowGraph] = useState(false);

  const nodes = proposal.graph?.nodes ?? [];
  const edges = proposal.graph?.edges ?? [];

  const handleApprove = async () => {
    setSubmitting("approve");
    try {
      await onDecide(true);
    } finally {
      setSubmitting(null);
    }
  };

  const handleReject = async () => {
    setSubmitting("reject");
    try {
      await onDecide(false, rejectReason.trim() || undefined);
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <div className="rounded-lg border border-[rgba(31,111,235,0.4)] bg-[rgba(31,111,235,0.08)] p-3">
      <div className="flex items-center gap-2 mb-2 text-sm font-medium text-[#fafafa]">
        <Sparkles className="h-4 w-4 text-[#5597ff]" />
        Workflow proposal
      </div>

      {proposal.rationale && (
        <p className="text-xs text-[#cfd3da] mb-3 leading-relaxed whitespace-pre-wrap">
          {proposal.rationale}
        </p>
      )}

      <div className="text-[11px] text-[#8c8c8d] mb-2">
        {nodes.length} node{nodes.length === 1 ? "" : "s"} · {edges.length} edge
        {edges.length === 1 ? "" : "s"}
        {proposal.pipelinesToLoad.length > 0 && (
          <>
            {" · loads "}
            <span className="font-mono text-[#b0b0b0]">
              {proposal.pipelinesToLoad.join(", ")}
            </span>
          </>
        )}
      </div>

      <button
        type="button"
        onClick={() => setShowGraph(v => !v)}
        className="text-[11px] text-[#5597ff] hover:underline mb-2"
      >
        {showGraph ? "Hide full graph" : "View full graph"}
      </button>
      {showGraph && (
        <pre className="text-[10px] text-[#b0b0b0] bg-[#0f0f0f] rounded px-2 py-1.5 border border-[rgba(255,255,255,0.04)] overflow-auto max-h-48 mb-2">
          {JSON.stringify(proposal.graph, null, 2)}
        </pre>
      )}

      {showRejectInput ? (
        <div className="space-y-2">
          <textarea
            value={rejectReason}
            onChange={e => setRejectReason(e.target.value)}
            placeholder="What should the agent try instead?"
            className="w-full text-xs bg-[#0f0f0f] border border-[rgba(255,255,255,0.08)] rounded px-2 py-1.5 text-[#e6e6e6]"
            rows={2}
          />
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowRejectInput(false)}
              disabled={submitting !== null}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={handleReject}
              disabled={submitting !== null}
            >
              Send rejection
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={handleApprove}
            disabled={submitting !== null}
            className="gap-1.5"
          >
            <Check className="h-3.5 w-3.5" />
            {submitting === "approve" ? "Approving…" : "Approve"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setShowRejectInput(true)}
            disabled={submitting !== null}
            className="gap-1.5"
          >
            <X className="h-3.5 w-3.5" />
            Reject
          </Button>
        </div>
      )}
    </div>
  );
}
