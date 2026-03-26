import { useState, useCallback } from "react";
import { MessageCircle, GraduationCap, Zap, ArrowLeft } from "lucide-react";
import { Button } from "../ui/button";
import { trackEvent } from "../../lib/analytics";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SurveyScreen = "intro" | "referral" | "use_case" | "onboarding_style";

export interface SurveyAnswers {
  referralSource: string | null;
  useCase: string | null;
  onboardingStyle: "teaching" | "simple";
}

interface CloudSurveyScreensProps {
  onComplete: (answers: SurveyAnswers) => void;
  /** Called when user presses back on the first screen. */
  onBack?: () => void;
}

// ---------------------------------------------------------------------------
// Option data
// ---------------------------------------------------------------------------

const REFERRAL_OPTIONS = ["Social Media", "Friends", "Search", "Other"];
const USE_CASE_OPTIONS = ["Personal", "Work"];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CloudSurveyScreens({ onComplete, onBack }: CloudSurveyScreensProps) {
  const [screen, setScreen] = useState<SurveyScreen>("intro");
  const [referralSource, setReferralSource] = useState<string | null>(null);
  const [useCase, setUseCase] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<
    "teaching" | "simple" | null
  >(null);
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
      trackEvent("onboarding_survey_answered", {
        screen: "referral",
        answer: option,
      });
      advanceAfterDelay("use_case");
    },
    [advanceAfterDelay]
  );

  const handleUseCaseSelect = useCallback(
    (option: string) => {
      setUseCase(option);
      trackEvent("onboarding_survey_answered", {
        screen: "use_case",
        answer: option,
      });
      advanceAfterDelay("onboarding_style");
    },
    [advanceAfterDelay]
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
      case "onboarding_style":
        setScreen("use_case");
        break;
    }
  }, [screen, onBack]);

  const handleSkip = useCallback(() => {
    trackEvent("onboarding_survey_skipped");
    setReferralSource(null);
    setUseCase(null);
    setScreen("onboarding_style");
  }, []);

  const handleStyleConfirm = useCallback(() => {
    if (!selectedStyle) return;
    trackEvent("onboarding_survey_answered", {
      screen: "onboarding_style",
      answer: selectedStyle,
    });
    onComplete({
      referralSource,
      useCase,
      onboardingStyle: selectedStyle,
    });
  }, [selectedStyle, referralSource, useCase, onComplete]);

  // Shared card wrapper — key on screen name to re-trigger entrance animation
  return (
    <div
      key={screen}
      className="w-full max-w-md mx-auto animate-in fade-in-0 slide-in-from-bottom-4 duration-500"
    >
      <div className="rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm p-6 space-y-5">
        {/* Back link — visible on all screens after intro */}
        {screen !== "intro" && (
          <button
            onClick={goBackSurvey}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors -mt-1 -ml-1"
          >
            <ArrowLeft className="h-3 w-3" />
            Back
          </button>
        )}

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
              While we connect to the cloud, let&rsquo;s get to know each other.
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

        {/* ---- Onboarding Style ---- */}
        {screen === "onboarding_style" && (
          <>
            <h3 className="text-sm font-medium text-foreground">
              Choose an onboarding style
            </h3>
            <div className="flex flex-col gap-3">
              <button
                onClick={() => setSelectedStyle("teaching")}
                className={`w-full flex items-start gap-3 p-4 rounded-lg border-2 text-left transition-all cursor-pointer ${
                  selectedStyle === "teaching"
                    ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                    : "border-border bg-card/50 hover:border-border/80 hover:bg-card"
                }`}
              >
                <div
                  className={`mt-0.5 flex items-center justify-center h-8 w-8 rounded-lg shrink-0 transition-colors ${
                    selectedStyle === "teaching"
                      ? "bg-foreground/10"
                      : "bg-muted"
                  }`}
                >
                  <GraduationCap className="h-4 w-4 text-foreground" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium text-foreground">
                    Teaching Mode
                  </p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Interactive onboarding, recommended for anyone already
                    familiar with node-based AI tools
                  </p>
                </div>
              </button>

              <button
                onClick={() => setSelectedStyle("simple")}
                className={`w-full flex items-start gap-3 p-4 rounded-lg border-2 text-left transition-all cursor-pointer ${
                  selectedStyle === "simple"
                    ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                    : "border-border bg-card/50 hover:border-border/80 hover:bg-card"
                }`}
              >
                <div
                  className={`mt-0.5 flex items-center justify-center h-8 w-8 rounded-lg shrink-0 transition-colors ${
                    selectedStyle === "simple" ? "bg-foreground/10" : "bg-muted"
                  }`}
                >
                  <Zap className="h-4 w-4 text-foreground" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium text-foreground">Simple</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Get a good result fast, recommended if you&rsquo;re new to
                    real-time AI
                  </p>
                </div>
              </button>
            </div>

            <Button
              onClick={handleStyleConfirm}
              disabled={!selectedStyle}
              className="w-full"
            >
              Continue
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
