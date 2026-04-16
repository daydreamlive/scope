import { ExternalLink } from "lucide-react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { useBilling } from "../contexts/BillingContext";
import { RedeemCodeSection } from "./settings/RedeemCodeSection";
import { setInferenceMode } from "../lib/onboardingStorage";
import { toast } from "sonner";

const TIERS = [
  {
    id: "pro" as const,
    name: "Pro",
    creditsPerMo: "500 credits/mo",
    hours: "~6 hrs on RTX 4090",
    description: "Great for getting started with regular creative sessions.",
    recommended: false,
  },
  {
    id: "max" as const,
    name: "Max",
    creditsPerMo: "1,750 credits/mo",
    hours: "~23 hrs on RTX 4090",
    description: "For creators who stream or iterate heavily every week.",
    recommended: true,
  },
];

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
  const handleSubscribe = (tierId: "pro" | "max") => {
    openCheckout(tierId);
    setShowPaywall(false);
  };

  const handleManageSubscription = () => {
    // Subscribed users are necessarily authenticated, so openCheckout here
    // just opens the dashboard without an auth detour.
    openCheckout("pro");
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

          {isSubscribed ? (
            <div>
              <Button
                className="w-full inline-flex items-center justify-center gap-1.5"
                onClick={handleManageSubscription}
              >
                Manage Subscription
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              {TIERS.map(planTier => (
                <div
                  key={planTier.id}
                  className={`relative flex flex-col p-5 rounded-lg border ${
                    planTier.recommended
                      ? "border-primary/60 bg-primary/5"
                      : "border-border"
                  }`}
                >
                  {planTier.recommended && (
                    <span className="absolute top-3 right-3 text-[10px] font-medium uppercase tracking-wider text-primary">
                      Recommended
                    </span>
                  )}
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-base font-semibold text-foreground">
                      {planTier.name}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {planTier.creditsPerMo}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {planTier.hours}
                  </p>
                  <p className="text-sm text-muted-foreground mt-3">
                    {planTier.description}
                  </p>
                  <button
                    onClick={() => handleSubscribe(planTier.id)}
                    className="mt-4 inline-flex items-center justify-center gap-1.5 h-10 rounded-full bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    Subscribe to {planTier.name}
                    <ExternalLink className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

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
