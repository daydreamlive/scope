import { useState } from "react";
import { X } from "lucide-react";
import { useBilling } from "../contexts/BillingContext";

const DISMISSED_KEY = "billing_transition_dismissed";

export function TransitionBanner() {
  const { tier, credits } = useBilling();
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(DISMISSED_KEY) === "true";
    } catch {
      return false;
    }
  });

  // Only show for users with welcome grant credits but no subscription
  const hasWelcomeCredits =
    tier === "free" && credits !== null && credits.balance > 0;

  if (dismissed || !hasWelcomeCredits) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(DISMISSED_KEY, "true");
    } catch {
      // ignore
    }
  };

  return (
    <div className="w-full bg-primary/10 border-b border-primary/20 px-4 py-2 flex items-center justify-between">
      <p className="text-xs text-foreground">
        Scope now uses credits for cloud inference. You've been granted{" "}
        <span className="font-medium">
          {Math.round(credits.balance)} free credits
        </span>
        .
      </p>
      <button
        onClick={handleDismiss}
        className="text-muted-foreground hover:text-foreground transition-colors ml-2"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
