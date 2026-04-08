import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { trackEvent } from "../lib/analytics";

type SeanEllisResponse =
  | "very_disappointed"
  | "somewhat_disappointed"
  | "not_disappointed";

interface SurveyModalProps {
  open: boolean;
  onClose: () => void;
}

const SEAN_ELLIS_OPTIONS: { label: string; value: SeanEllisResponse }[] = [
  { label: "Very disappointed", value: "very_disappointed" },
  { label: "Somewhat disappointed", value: "somewhat_disappointed" },
  { label: "Not disappointed", value: "not_disappointed" },
];

export function SurveyModal({ open, onClose }: SurveyModalProps) {
  const [step, setStep] = useState<"sean_ellis" | "nps">("sean_ellis");

  const handleSeanEllisSelect = (value: SeanEllisResponse) => {
    trackEvent("sean_ellis_response", { response: value });
    setStep("nps");
  };

  const handleNpsSelect = (score: number) => {
    trackEvent("nps_response", { score });
    onClose();
  };

  const handleSkip = () => {
    onClose();
  };

  // Reset step when modal reopens (shouldn't happen, but defensive)
  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        {step === "sean_ellis" ? (
          <>
            <DialogHeader>
              <DialogTitle>Quick question</DialogTitle>
              <DialogDescription>
                How would you feel if you could no longer use Daydream Scope?
              </DialogDescription>
            </DialogHeader>
            <div className="flex flex-col gap-2 mt-2">
              {SEAN_ELLIS_OPTIONS.map(({ label, value }) => (
                <Button
                  key={value}
                  variant="outline"
                  className="justify-start text-left h-auto py-3 px-4"
                  onClick={() => handleSeanEllisSelect(value)}
                >
                  {label}
                </Button>
              ))}
            </div>
            <div className="flex justify-end mt-1">
              <button
                onClick={handleSkip}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors underline-offset-2 hover:underline"
              >
                Skip
              </button>
            </div>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>One more</DialogTitle>
              <DialogDescription>
                How likely are you to recommend Daydream Scope to a friend or
                colleague?
              </DialogDescription>
            </DialogHeader>
            <div className="mt-2">
              <div className="flex gap-1 flex-wrap justify-center">
                {Array.from({ length: 11 }, (_, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    className="w-10 h-10 p-0 text-sm font-medium"
                    onClick={() => handleNpsSelect(i)}
                  >
                    {i}
                  </Button>
                ))}
              </div>
              <div className="flex justify-between mt-1 px-1">
                <span className="text-xs text-muted-foreground">
                  Not likely
                </span>
                <span className="text-xs text-muted-foreground">
                  Very likely
                </span>
              </div>
            </div>
            <div className="flex justify-end mt-1">
              <button
                onClick={handleSkip}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors underline-offset-2 hover:underline"
              >
                Skip
              </button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
