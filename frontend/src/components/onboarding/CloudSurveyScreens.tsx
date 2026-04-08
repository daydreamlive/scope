import { useState, useCallback } from "react";
import { MessageCircle, ArrowLeft } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SurveyScreen = "intro" | "referral" | "use_case";

export interface SurveyAnswers {
  referralSource: string | null;
  useCase: string | null;
}

interface CloudSurveyScreensProps {
  onComplete: (answers: SurveyAnswers) => void;
  /** Called when user presses back on the first screen. */
  onBack?: () => void;
  /** Start at a specific screen (default: "intro") */
  initialScreen?: SurveyScreen;
}

// ---------------------------------------------------------------------------
// Option data
// ---------------------------------------------------------------------------

const REFERRAL_OPTIONS = ["Social Media", "Friends", "Search", "Other"];
const USE_CASE_OPTIONS = ["Personal", "Work"];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CloudSurveyScreens({
  onComplete,
  onBack,
  initialScreen = "intro",
}: CloudSurveyScreensProps) {
  const [screen, setScreen] = useState<SurveyScreen>(initialScreen);
  const [referralSource, setReferralSource] = useState<string | null>(null);
  const [useCase, setUseCase] = useState<string | null>(null);
  const [transitioning, setTransitioning] = useState(false);

  // Advance with a brief delay so the user sees their selection highlight
  const advanceAfterDelay = useCallback((next: SurveyScreen, delayMs = 300) => {
    setTransitioning(true);
    setTimeout(() => {
      setScreen(next);
      setTransitioning(false);
    }, delayMs);
  }, []);

  const handleReferralSelect = useCallback(
    (option: string) => {
      setReferralSource(option);
      advanceAfterDelay("use_case");
    },
    [advanceAfterDelay]
  );

  const handleUseCaseSelect = useCallback(
    (option: string) => {
      const newUseCase = option;
      setUseCase(newUseCase);
      setTransitioning(true);
      setTimeout(() => {
        onComplete({ referralSource, useCase: newUseCase });
        setTransitioning(false);
      }, 300);
    },
    [referralSource, onComplete]
  );

  const goBackSurvey = useCallback(() => {
    switch (screen) {
      case "intro":
        onBack?.();
        break;
      case "referral":
        setScreen("intro");
        break;
      case "use_case":
        setScreen("referral");
        break;
    }
  }, [screen, onBack]);

  const handleSkip = useCallback(() => {
    onComplete({ referralSource: null, useCase: null });
  }, [onComplete]);

  // Shared card wrapper — key on screen name to re-trigger entrance animation
  return (
    <>
      {/* Back button — fixed top-left to match the overlay's back button style */}
      <button
        onClick={goBackSurvey}
        className="fixed top-6 left-6 z-[110] flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </button>

      <div
        key={screen}
        className="w-full max-w-md mx-auto animate-in fade-in-0 slide-in-from-bottom-4 duration-500"
      >
        <div className="rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm p-6 space-y-5">
          {/* ---- Intro ---- */}
          {screen === "intro" && (
            <>
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-muted">
                  <MessageCircle className="h-4 w-4 text-muted-foreground" />
                </div>
                <h3 className="text-sm font-medium text-foreground">
                  Getting Started
                </h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                While we connect to the cloud, let&rsquo;s get to know each
                other.
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleSkip}
                  className="flex-1 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted/50 transition-colors text-foreground"
                >
                  No thanks
                </button>
                <button
                  onClick={() => setScreen("referral")}
                  className="flex-1 px-4 py-2 text-sm font-medium rounded-lg bg-foreground text-background hover:bg-foreground/90 transition-colors"
                >
                  Next
                </button>
              </div>
            </>
          )}

          {/* ---- Referral Source ---- */}
          {screen === "referral" && (
            <>
              <h3 className="text-sm font-medium text-foreground">
                How&rsquo;d you hear about Daydream?
              </h3>
              <div className="flex flex-col gap-2">
                {REFERRAL_OPTIONS.map(option => (
                  <button
                    key={option}
                    onClick={() => handleReferralSelect(option)}
                    disabled={transitioning}
                    className={`w-full px-4 py-3 text-sm font-medium rounded-lg border text-left transition-all ${
                      referralSource === option
                        ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                        : "border-border hover:border-border/80 hover:bg-muted/50"
                    } text-foreground`}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </>
          )}

          {/* ---- Use Case ---- */}
          {screen === "use_case" && (
            <>
              <h3 className="text-sm font-medium text-foreground">
                Are you using Daydream for work?
              </h3>
              <div className="flex flex-col gap-2">
                {USE_CASE_OPTIONS.map(option => (
                  <button
                    key={option}
                    onClick={() => handleUseCaseSelect(option)}
                    disabled={transitioning}
                    className={`w-full px-4 py-3 text-sm font-medium rounded-lg border text-left transition-all ${
                      useCase === option
                        ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                        : "border-border hover:border-border/80 hover:bg-muted/50"
                    } text-foreground`}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
