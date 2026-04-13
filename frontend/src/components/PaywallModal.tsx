import { useState } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { useBilling } from "../contexts/BillingContext";
import { redeemCreditCode } from "../lib/billing";
import { getDaydreamAPIKey } from "../lib/auth";
import { toast } from "sonner";

const TIERS = [
  {
    id: "pro" as const,
    name: "Pro",
    price: "$10/mo",
    credits: "500 credits",
    recommended: false,
  },
  {
    id: "max" as const,
    name: "Max",
    price: "$30/mo",
    credits: "1,750 credits",
    recommended: true,
  },
];

function getHeadline(
  reason: "trial_exhausted" | "credits_exhausted" | "subscribe" | null,
): string {
  switch (reason) {
    case "trial_exhausted":
      return "Your free remote inference time is up";
    case "credits_exhausted":
      return "You've run out of credits";
    case "subscribe":
      return "Choose a plan";
    default:
      return "Subscribe to continue";
  }
}

export function PaywallModal() {
  const {
    showPaywall,
    setShowPaywall,
    paywallReason,
    openCheckout,
    toggleOverage,
    subscription,
    creditsPerMin,
    refresh,
  } = useBilling();

  const [redeemCode, setRedeemCode] = useState("");
  const [isRedeeming, setIsRedeeming] = useState(false);

  const handleTierClick = (tier: "pro" | "max") => {
    openCheckout(tier);
    setShowPaywall(false);
  };

  const handleOverage = () => {
    toggleOverage(true);
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
        `${result.credits} credits added${result.label ? ` — ${result.label}` : ""}`,
      );
      setRedeemCode("");
      refresh();
      setShowPaywall(false);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to redeem code",
      );
    } finally {
      setIsRedeeming(false);
    }
  };

  const rateDisplay =
    creditsPerMin > 0
      ? `Your current workflow uses ${creditsPerMin} credits/min.`
      : "Credit usage varies by workflow and GPU type.";

  return (
    <Dialog
      open={showPaywall}
      onOpenChange={open => !open && setShowPaywall(false)}
    >
      <DialogContent className="sm:max-w-[480px]">
        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              {getHeadline(paywallReason)}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">{rateDisplay}</p>
          </div>

          <div className="grid gap-3">
            {TIERS.map(tier => (
              <button
                key={tier.id}
                onClick={() => handleTierClick(tier.id)}
                className={`flex items-center justify-between p-4 rounded-lg border text-left transition-colors hover:bg-muted/50 ${
                  tier.recommended
                    ? "border-primary bg-primary/5"
                    : "border-border"
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-foreground">
                      {tier.name}
                    </span>
                    {tier.recommended && (
                      <span className="text-[10px] font-medium uppercase tracking-wider text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                        Recommended
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    {tier.credits}
                  </p>
                </div>
                <span className="text-sm font-medium text-foreground">
                  {tier.price}
                </span>
              </button>
            ))}
          </div>

          {/* Overage option — only show when credits exhausted and user has subscription */}
          {paywallReason === "credits_exhausted" && subscription && (
            <Button
              variant="outline"
              className="w-full"
              onClick={handleOverage}
            >
              Enable overage — 500 credits for $10 (recurring when depleted)
            </Button>
          )}

          {/* Redeem code — show for trial exhausted or credits exhausted */}
          {(paywallReason === "trial_exhausted" ||
            paywallReason === "credits_exhausted") && (
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
            <button
              onClick={() => setShowPaywall(false)}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Maybe later
            </button>
          </div>

          <p className="text-xs text-center text-muted-foreground">
            Questions?{" "}
            <a
              href="mailto:support@daydream.monster"
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
