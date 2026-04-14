import { useState } from "react";
import { useBilling } from "../../contexts/BillingContext";
import { redeemCreditCode } from "../../lib/billing";
import { getDaydreamAPIKey } from "../../lib/auth";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import { toast } from "sonner";

function RedeemCodeSection({ onRedeemed }: { onRedeemed: () => void }) {
  const [code, setCode] = useState("");
  const [isRedeeming, setIsRedeeming] = useState(false);

  const handleRedeem = async () => {
    const trimmed = code.trim();
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
      setCode("");
      onRedeemed();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to redeem code");
    } finally {
      setIsRedeeming(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-foreground">Redeem Code</div>
      <div className="flex gap-2">
        <input
          type="text"
          value={code}
          onChange={e => setCode(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === "Enter" && handleRedeem()}
          placeholder="DD-XXXX-XXXX"
          className="flex-1 h-8 rounded-md border border-input bg-background px-3 text-sm font-mono placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          disabled={isRedeeming}
        />
        <Button
          size="sm"
          onClick={handleRedeem}
          disabled={!code.trim() || isRedeeming}
        >
          {isRedeeming ? "Redeeming..." : "Redeem"}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Enter a credit code to add credits to your balance.
      </p>
    </div>
  );
}

export function BillingTab() {
  const {
    tier,
    credits,
    subscription,
    creditsPerMin,
    toggleOverage,
    refresh,
    openCheckout,
  } = useBilling();

  const [showOverageConfirm, setShowOverageConfirm] = useState(false);

  const handleSubscribe = () => {
    openCheckout("pro");
  };

  const estimatedMinutes =
    credits && creditsPerMin > 0 && credits.balance > 0
      ? Math.round(credits.balance / creditsPerMin)
      : null;

  if (tier === "free") {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-foreground">
          Subscription & Billing
        </h3>
        <p className="text-sm text-muted-foreground">
          You're on the free plan. Subscribe to get credits for Daydream
          Cloud.
        </p>
        {credits && credits.balance > 0 && (
          <div className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">
              {Math.round(credits.balance)}
            </span>{" "}
            welcome credits remaining
            {estimatedMinutes !== null && (
              <span className="text-muted-foreground">
                {" "}
                (~{estimatedMinutes} min)
              </span>
            )}
          </div>
        )}
        <Button size="sm" onClick={handleSubscribe}>
          Subscribe
        </Button>
        <div className="pt-2 border-t border-border">
          <RedeemCodeSection onRedeemed={refresh} />
        </div>
        <p className="text-xs text-muted-foreground">
          Questions about billing?{" "}
          <a
            href="mailto:support@daydream.monster"
            className="underline hover:text-foreground transition-colors"
          >
            Contact support
          </a>
        </p>
      </div>
    );
  }

  const tierLabel = tier === "pro" ? "Pro" : "Max";
  const tierPrice = tier === "pro" ? "$10/mo" : "$30/mo";
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
            {estimatedMinutes !== null && (
              <span className="text-muted-foreground">
                {" "}
                (~{estimatedMinutes} min remaining)
              </span>
            )}
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
          {creditsPerMin > 0 && (
            <p className="text-xs text-muted-foreground">
              Current rate: {creditsPerMin} credits/min
            </p>
          )}
        </div>
      )}

      {/* Overage toggle */}
      <div className="flex items-start gap-3">
        <Switch
          checked={subscription?.overageEnabled ?? false}
          onCheckedChange={checked => {
            if (checked) {
              setShowOverageConfirm(true);
            } else {
              toggleOverage(false);
            }
          }}
          className="mt-0.5"
        />
        <div>
          <div className="text-sm font-medium text-foreground">
            Overage billing
          </div>
          <p className="text-xs text-muted-foreground">
            When your monthly credits run out, automatically add 500 credits for
            $10 (up to 5 times per cycle, $50 max).
          </p>
        </div>
      </div>

      {/* Overage confirmation dialog */}
      <AlertDialog
        open={showOverageConfirm}
        onOpenChange={setShowOverageConfirm}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Enable overage billing?</AlertDialogTitle>
            <AlertDialogDescription>
              When your monthly credits run out, you'll be automatically charged{" "}
              <span className="font-medium text-foreground">
                $10 for 500 additional credits
              </span>
              . This can happen up to{" "}
              <span className="font-medium text-foreground">
                5 times per billing cycle ($50 max)
              </span>
              . You can disable this anytime in Settings.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                toggleOverage(true);
                setShowOverageConfirm(false);
              }}
            >
              Enable Overage
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 pt-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            window.open(
              "https://app.daydream.monster/dashboard/usage",
              "_blank"
            )
          }
        >
          Manage Subscription
        </Button>
      </div>

      {/* Redeem code */}
      <div className="pt-2 border-t border-border">
        <RedeemCodeSection onRedeemed={refresh} />
      </div>

      {/* Help & support */}
      <p className="text-xs text-muted-foreground">
        Questions about billing?{" "}
        <a
          href="mailto:support@daydream.monster"
          className="underline hover:text-foreground transition-colors"
        >
          Contact support
        </a>
        {" · "}
        <a
          href="https://docs.daydream.monster/billing"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-foreground transition-colors"
        >
          How credits work
        </a>
      </p>
    </div>
  );
}
