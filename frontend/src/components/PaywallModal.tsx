import { useState } from "react";
import { ExternalLink } from "lucide-react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { useBilling } from "../contexts/BillingContext";
import { redeemCreditCode } from "../lib/billing";
import { getDaydreamAPIKey } from "../lib/auth";
import { openExternalUrl } from "../lib/openExternal";
import { toast } from "sonner";

const DASHBOARD_USAGE_URL = "https://app.daydream.live/dashboard/usage";

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
    return "To continue generating, please purchase additional credits or enable auto-top-up.";
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
  } = useBilling();

  const [redeemCode, setRedeemCode] = useState("");
  const [isRedeeming, setIsRedeeming] = useState(false);

  const isSubscribed = tier === "pro" || tier === "max";

  const handleSubscribe = (_tierId: "pro" | "max") => {
    openExternalUrl(DASHBOARD_USAGE_URL);
    setShowPaywall(false);
  };

  const handleManageSubscription = () => {
    openExternalUrl(DASHBOARD_USAGE_URL);
    setShowPaywall(false);
  };

  const handleRunLocally = async () => {
    try {
      await fetch("/api/v1/cloud/disconnect", { method: "POST" });
      toast.info("Switched to local inference");
    } catch {
      // Cloud may already be disconnected
    }
    setShowPaywall(false);
  };

  const handleRedeem = async () => {
    const trimmed = redeemCode.trim();
    if (!trimmed) return;

    setIsRedeeming(true);
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        toast.error("Please sign in to redeem a code");
        return;
      }
      const result = await redeemCreditCode(apiKey, trimmed);
      toast.success(
        `${result.credits} credits added${result.label ? ` — ${result.label}` : ""}`
      );
      setRedeemCode("");
      refresh();
      setShowPaywall(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to redeem code");
    } finally {
      setIsRedeeming(false);
    }
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
              {TIERS.map(tier => (
                <div
                  key={tier.id}
                  className={`relative flex flex-col p-5 rounded-lg border ${
                    tier.recommended
                      ? "border-primary/60 bg-primary/5"
                      : "border-border"
                  }`}
                >
                  {tier.recommended && (
                    <span className="absolute top-3 right-3 text-[10px] font-medium uppercase tracking-wider text-primary">
                      Recommended
                    </span>
                  )}
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-base font-semibold text-foreground">
                      {tier.name}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {tier.creditsPerMo}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {tier.hours}
                  </p>
                  <p className="text-sm text-muted-foreground mt-3">
                    {tier.description}
                  </p>
                  <button
                    onClick={() => handleSubscribe(tier.id)}
                    className="mt-4 inline-flex items-center justify-center gap-1.5 h-10 rounded-full bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    Subscribe to {tier.name}
                    <ExternalLink className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Redeem code — show when credits exhausted */}
          {paywallReason === "credits_exhausted" && (
            <div className="flex gap-2">
              <input
                type="text"
                value={redeemCode}
                onChange={e => setRedeemCode(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === "Enter" && handleRedeem()}
                placeholder="Have a code? DD-XXXX-XXXX"
                className="flex-1 h-8 rounded-md border border-input bg-background px-3 text-sm font-mono placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                disabled={isRedeeming}
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleRedeem}
                disabled={!redeemCode.trim() || isRedeeming}
              >
                {isRedeeming ? "..." : "Redeem"}
              </Button>
            </div>
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
