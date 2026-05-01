import { useState } from "react";
import { redeemCreditCode } from "../../lib/billing";
import { getDaydreamAPIKey } from "../../lib/auth";
import { Button } from "../ui/button";
import { toast } from "sonner";

export function RedeemCodeSection({ onRedeemed }: { onRedeemed: () => void }) {
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
