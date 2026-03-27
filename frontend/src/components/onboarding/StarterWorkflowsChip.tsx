import { useState, useMemo } from "react";
import { X } from "lucide-react";
import { getWorkflowsForStyle } from "./starterWorkflows";

const LS_KEY = "scope_starter_chip_dismissed";
const CHOSEN_KEY = "scope_starter_chosen_id";

interface StarterWorkflowsChipProps {
  /** Called when the user clicks the chip. Should open the workflows tab. */
  onOpenWorkflows: () => void;
}

/**
 * A floating chip in the bottom-right of the canvas that shows an untried
 * starter workflow thumbnail and invites the user to explore more.
 * Dismissible and persisted.
 */
export function StarterWorkflowsChip({
  onOpenWorkflows,
}: StarterWorkflowsChipProps) {
  const [dismissed, setDismissed] = useState(
    () => !!localStorage.getItem(LS_KEY)
  );
  // Always show simple-mode workflows — teaching ones have notes/missing source
  const starters = getWorkflowsForStyle("simple");

  // Pick one workflow the user hasn't tried yet
  const suggestion = useMemo(() => {
    const chosenId = localStorage.getItem(CHOSEN_KEY);
    const untried = starters.filter(wf => wf.id !== chosenId);
    return untried.length > 0 ? untried[0] : null;
  }, [starters]);

  if (dismissed || !suggestion) return null;

  return (
    <div className="fixed bottom-6 right-6 z-40 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
      <div className="flex items-stretch bg-[#1a1a1a] border border-[rgba(119,119,119,0.2)] rounded-xl shadow-lg overflow-hidden max-w-[280px]">
        {/* Thumbnail */}
        <button
          onClick={onOpenWorkflows}
          className="relative w-20 shrink-0 overflow-hidden"
        >
          <img
            src={suggestion.thumbnail}
            alt={suggestion.title}
            className="h-full w-full object-cover"
          />
        </button>

        {/* Text content */}
        <button
          onClick={onOpenWorkflows}
          className="flex-1 p-3 text-left hover:bg-[rgba(255,255,255,0.03)] transition-colors"
        >
          <p className="text-xs font-medium text-[#fafafa] leading-tight">
            Try more starter workflows
          </p>
        </button>

        {/* Dismiss */}
        <button
          onClick={e => {
            e.stopPropagation();
            setDismissed(true);
            localStorage.setItem(LS_KEY, "1");
          }}
          className="self-start p-1.5 m-1 text-[#666] hover:text-[#aaa] transition-colors rounded-full hover:bg-[rgba(255,255,255,0.05)]"
          aria-label="Dismiss"
        >
          <X className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
