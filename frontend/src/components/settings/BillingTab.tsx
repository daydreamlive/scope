import { useBilling } from "../../contexts/BillingContext";
import { Button } from "../ui/button";

export function BillingTab() {
  const {
    tier,
    credits,
    subscription,
    openCheckout,
    openPortal,
    toggleOverage,
    setShowPaywall,
    setPaywallReason,
  } = useBilling();

  const handleSubscribe = () => {
    setPaywallReason("subscribe");
    setShowPaywall(true);
  };

  if (tier === "free") {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-foreground">
          Subscription & Billing
        </h3>
        <p className="text-sm text-muted-foreground">
          You're on the free plan. Subscribe to get credits for remote
          inference.
        </p>
        {credits && credits.balance > 0 && (
          <div className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">
              {Math.round(credits.balance)}
            </span>{" "}
            welcome credits remaining
          </div>
        )}
        <Button size="sm" onClick={handleSubscribe}>
          Subscribe
        </Button>
      </div>
    );
  }

  const tierLabel = tier === "basic" ? "Basic" : "Pro";
  const tierPrice = tier === "basic" ? "$10/mo" : "$30/mo";
  const renewDate = subscription?.currentPeriodEnd
    ? new Date(subscription.currentPeriodEnd).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      })
    : "—";

  return (
    <div className="space-y-5">
      <h3 className="text-sm font-medium text-foreground">
        Subscription & Billing
      </h3>

      {/* Plan info */}
      <div className="space-y-1">
        <div className="text-sm">
          <span className="font-medium text-foreground">{tierLabel}</span>
          <span className="text-muted-foreground"> — {tierPrice}</span>
          {subscription?.cancelAtPeriodEnd && (
            <span className="text-amber-500 ml-2 text-xs">
              Cancels {renewDate}
            </span>
          )}
          {!subscription?.cancelAtPeriodEnd && (
            <span className="text-muted-foreground ml-2 text-xs">
              Renews {renewDate}
            </span>
          )}
        </div>
      </div>

      {/* Credits */}
      {credits && (
        <div className="space-y-1">
          <div className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">
              {Math.round(credits.balance)}
            </span>{" "}
            of {Math.round(credits.periodCredits)} credits
          </div>
          <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                credits.balance > credits.periodCredits * 0.2
                  ? "bg-green-500"
                  : credits.balance > credits.periodCredits * 0.05
                    ? "bg-amber-400"
                    : "bg-red-500"
              }`}
              style={{
                width: `${Math.min(100, (credits.balance / credits.periodCredits) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Overage toggle */}
      <div className="flex items-start gap-3">
        <label className="relative inline-flex items-center cursor-pointer mt-0.5">
          <input
            type="checkbox"
            className="sr-only peer"
            checked={subscription?.overageEnabled ?? false}
            onChange={e => {
              if (e.target.checked) {
                // Confirm before enabling — users should understand the cost
                const ok = window.confirm(
                  "Enable overage billing?\n\nWhen your monthly credits run out, you'll be automatically charged $10 for 300 additional credits. This can happen up to 5 times per billing cycle ($50 max).\n\nYou can disable this anytime in Settings.",
                );
                if (!ok) return;
              }
              toggleOverage(e.target.checked);
            }}
          />
          <div className="w-9 h-5 bg-muted rounded-full peer peer-checked:bg-primary transition-colors after:content-[''] after:absolute after:top-0.5 after:start-[2px] after:bg-background after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-full" />
        </label>
        <div>
          <div className="text-sm font-medium text-foreground">
            Overage billing
          </div>
          <p className="text-xs text-muted-foreground">
            When your monthly credits run out, automatically add 300 credits for
            $10.
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 pt-2">
        <Button variant="outline" size="sm" onClick={openPortal}>
          Manage Subscription
        </Button>
        {tier === "basic" && (
          <Button size="sm" onClick={() => openCheckout("pro")}>
            Upgrade to Pro
          </Button>
        )}
      </div>
    </div>
  );
}
