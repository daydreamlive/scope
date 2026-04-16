import { useEffect, type ReactNode } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { trackEvent } from "../lib/analytics";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Shared shell
// ---------------------------------------------------------------------------

type ShellSize = "md" | "lg";

interface SurveyShellProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description: string;
  size?: ShellSize;
  /** Event name to fire when the modal opens. */
  shownEvent?: string;
  /** Event name to fire when the user skips/dismisses without answering. */
  skippedEvent?: string;
  children: ReactNode;
}

function SurveyShell({
  open,
  onClose,
  title,
  description,
  size = "md",
  shownEvent,
  skippedEvent,
  children,
}: SurveyShellProps) {
  // Fire _shown on open so we have a denominator for response rates.
  useEffect(() => {
    if (open && shownEvent) {
      trackEvent(shownEvent);
    }
  }, [open, shownEvent]);

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      if (skippedEvent) trackEvent(skippedEvent);
      onClose();
    }
  };

  const handleSkip = () => {
    if (skippedEvent) trackEvent(skippedEvent);
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className={cn(
          "overflow-hidden",
          size === "lg" ? "max-w-lg" : "max-w-md"
        )}
      >
        {/* Brand "fog" aurora — static CSS version of the onboarding
            FogOfWarBackground effect. Subtle, non-interactive. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 -z-10"
          style={{
            background: `
              radial-gradient(ellipse 80% 60% at 85% 10%, rgba(255, 152, 46, 0.10) 0%, transparent 60%),
              radial-gradient(ellipse 70% 60% at 15% 90%, rgba(47, 190, 197, 0.09) 0%, transparent 60%),
              radial-gradient(ellipse 60% 50% at 50% 50%, rgba(247, 59, 65, 0.05) 0%, transparent 65%)
            `,
          }}
        />
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        {children}
        <div className="flex justify-end mt-1">
          <button
            onClick={handleSkip}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors underline-offset-2 hover:underline"
          >
            Skip
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Sean Ellis (vertical / stacked)
// ---------------------------------------------------------------------------

type SeanEllisResponse =
  | "very_disappointed"
  | "somewhat_disappointed"
  | "not_disappointed";

const SEAN_ELLIS_OPTIONS: { label: string; value: SeanEllisResponse }[] = [
  { label: "Very disappointed", value: "very_disappointed" },
  { label: "Somewhat disappointed", value: "somewhat_disappointed" },
  { label: "Not disappointed", value: "not_disappointed" },
];

interface SurveyProps {
  open: boolean;
  onClose: () => void;
}

export function SeanEllisSurvey({ open, onClose }: SurveyProps) {
  const handleSelect = (value: SeanEllisResponse) => {
    trackEvent("sean_ellis_response", { response: value });
    onClose();
  };

  return (
    <SurveyShell
      open={open}
      onClose={onClose}
      title="Help us improve Scope"
      description="How would you feel if you could no longer use Daydream Scope?"
      size="md"
      shownEvent="sean_ellis_shown"
      skippedEvent="sean_ellis_skipped"
    >
      <div
        role="radiogroup"
        aria-label="Disappointment level"
        className="flex flex-col mt-2"
      >
        {SEAN_ELLIS_OPTIONS.map(({ label, value }, i) => {
          // Brand gradient mapping: "Very disappointed" is the strong PMF
          // signal → teal (positive end), "Not disappointed" → red (negative
          // end), middle → orange. Matches the NPS scale ordering and uses
          // the onboarding fog-of-war palette. Applied on hover only so
          // default state doesn't bias the user.
          const bucketHover =
            i === 0
              ? "hover:bg-[#2FBEC5]/15 hover:border-[#2FBEC5]/25"
              : i === 1
                ? "hover:bg-[#FF982E]/15 hover:border-[#FF982E]/25"
                : "hover:bg-[#F73B41]/15 hover:border-[#F73B41]/25";
          // Connect buttons vertically: first keeps its top radius, last
          // keeps its bottom radius, middle collapses.
          const radius =
            i === 0
              ? "rounded-t-md rounded-b-none"
              : i === SEAN_ELLIS_OPTIONS.length - 1
                ? "rounded-b-md rounded-t-none"
                : "rounded-none";
          return (
            <Button
              key={value}
              role="radio"
              aria-checked={false}
              variant="outline"
              className={cn(
                "justify-start text-left h-11 px-4 -mt-px first:mt-0 focus:z-10",
                radius,
                bucketHover
              )}
              onClick={() => handleSelect(value)}
            >
              {label}
            </Button>
          );
        })}
      </div>
    </SurveyShell>
  );
}

// ---------------------------------------------------------------------------
// NPS (horizontal / traditional)
// ---------------------------------------------------------------------------

export function NpsSurvey({ open, onClose }: SurveyProps) {
  const handleSelect = (score: number) => {
    trackEvent("nps_response", { score });
    onClose();
  };

  return (
    <SurveyShell
      open={open}
      onClose={onClose}
      title="Help us improve Scope"
      description="How likely are you to recommend Daydream Scope to a friend or colleague?"
      size="lg"
      shownEvent="nps_shown"
      skippedEvent="nps_skipped"
    >
      <div className="mt-2">
        <div
          role="radiogroup"
          aria-label="Net Promoter Score, 0 to 10"
          aria-describedby="nps-anchors"
          className="flex flex-nowrap"
        >
          {Array.from({ length: 11 }, (_, i) => {
            // Brand gradient mapping: promoters (9-10) → teal (positive end),
            // passives (7-8) → orange, detractors (0-6) → red. Uses the
            // onboarding fog-of-war palette. Applied on hover only so the
            // default state doesn't bias the user.
            const bucketHover =
              i >= 9
                ? "hover:bg-[#2FBEC5]/15 hover:border-[#2FBEC5]/25"
                : i >= 7
                  ? "hover:bg-[#FF982E]/15 hover:border-[#FF982E]/25"
                  : "hover:bg-[#F73B41]/15 hover:border-[#F73B41]/25";
            // Connect buttons: only the first keeps its left radius,
            // only the last keeps its right radius.
            const radius =
              i === 0
                ? "rounded-l-md rounded-r-none"
                : i === 10
                  ? "rounded-r-md rounded-l-none"
                  : "rounded-none";
            return (
              <Button
                key={i}
                role="radio"
                aria-checked={false}
                variant="outline"
                className={cn(
                  "flex-1 min-w-0 h-11 p-0 text-sm font-medium -ml-px first:ml-0 focus:z-10",
                  radius,
                  bucketHover
                )}
                onClick={() => handleSelect(i)}
              >
                {i}
              </Button>
            );
          })}
        </div>
        <div
          id="nps-anchors"
          className="flex justify-between mt-1 px-1 text-xs text-muted-foreground"
        >
          <span>Not likely</span>
          <span>Very likely</span>
        </div>
      </div>
    </SurveyShell>
  );
}
