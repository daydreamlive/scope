import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { useBilling } from "../contexts/BillingContext";

const TIERS = [
  {
    id: "basic" as const,
    name: "Basic",
    price: "$10/mo",
    credits: "500 credits",
    recommended: false,
  },
  {
    id: "pro" as const,
    name: "Pro",
    price: "$30/mo",
    credits: "1,750 credits",
    recommended: true,
  },
];

function getHeadline(
  reason: "trial_exhausted" | "credits_exhausted" | "subscribe" | null
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
  } = useBilling();

  const handleTierClick = (tier: "basic" | "pro") => {
    openCheckout(tier);
    setShowPaywall(false);
  };

  const handleOverage = () => {
    toggleOverage(true);
    setShowPaywall(false);
  };

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
            <p className="text-sm text-muted-foreground mt-1">
              Credits are used at different rates depending on your workflow.
              Most workflows use 7.5 credits/min.
            </p>
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
              Add 300 credits for $10
            </Button>
          )}

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <button
              onClick={() => setShowPaywall(false)}
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
        </div>
      </DialogContent>
    </Dialog>
  );
}
