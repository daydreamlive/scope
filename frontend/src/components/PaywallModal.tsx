import { ExternalLink } from "lucide-react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { useBilling } from "../contexts/BillingContext";
import { RedeemCodeSection } from "./settings/RedeemCodeSection";
import { setInferenceMode } from "../lib/onboardingStorage";
import { toast } from "sonner";

function getHeadline(
  reason: "credits_exhausted" | "subscribe" | null,
  isSubscribed: boolean
): string {
  if (isSubscribed) return "You've run out of credits";
  switch (reason) {
    case "credits_exhausted":
      return "You've run out of credits";
    case "subscribe":
      return "Choose a plan";
    default:
      return "Subscribe to continue";
  }
}

function getSubcopy(
  reason: "credits_exhausted" | "subscribe" | null,
  isSubscribed: boolean
): string {
  if (isSubscribed) {
    // Subscribed users only see the Manage Subscription CTA below — the copy
    // must match, otherwise it promises options the modal doesn't expose.
    return "Manage your subscription to top up credits or enable overage billing.";
  }
  switch (reason) {
    case "credits_exhausted":
      return "To continue generating, please choose a subscription.";
    default:
      return "Choose a plan to continue generating.";
  }
}

export function PaywallModal() {
  const {
    showPaywall,
    setShowPaywall,
    paywallReason,
    tier,
    refresh,
    openCheckout,
  } = useBilling();

  const isSubscribed = tier === "pro" || tier === "max";

  // Route through BillingContext.openCheckout so unauthenticated users are
  // sent through the sign-in flow before landing on the billing page.
  const handleSubscribe = () => {
    openCheckout();
    setShowPaywall(false);
  };

  const handleRunLocally = async () => {
    // Persist the inference-mode switch first so the app won't auto-reconnect
    // to cloud on next launch. This is independent of the disconnect call,
    // which only tears down the current session.
    try {
      await setInferenceMode("local");
    } catch {
      // persistence failures are already swallowed inside the helper
    }
    try {
      await fetch("/api/v1/cloud/disconnect", { method: "POST" });
      toast.info("Switched to local inference");
    } catch {
      // Cloud may already be disconnected — still close the paywall
    }
    setShowPaywall(false);
  };

  return (
    <Dialog
      open={showPaywall}
      onOpenChange={open => !open && setShowPaywall(false)}
    >
      <DialogContent className="sm:max-w-[720px]">
        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              {getHeadline(paywallReason, isSubscribed)}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {getSubcopy(paywallReason, isSubscribed)}
            </p>
          </div>

          <div>
            <Button
              className="w-full inline-flex items-center justify-center gap-1.5"
              onClick={handleSubscribe}
            >
              {isSubscribed ? "Manage Subscription" : "Choose a plan"}
              <ExternalLink className="h-4 w-4" />
            </Button>
          </div>

          {/* Redeem code — show when credits exhausted */}
          {paywallReason === "credits_exhausted" && (
            <RedeemCodeSection
              onRedeemed={() => {
                refresh();
                setShowPaywall(false);
              }}
            />
          )}

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <button
              onClick={handleRunLocally}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Run locally instead
            </button>
          </div>

          <p className="text-xs text-center text-muted-foreground">
            Questions?{" "}
            <a
              href="mailto:support@daydream.live"
              className="underline hover:text-foreground transition-colors"
            >
              Contact support
            </a>
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
