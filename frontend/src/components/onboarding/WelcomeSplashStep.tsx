interface WelcomeSplashStepProps {
  onAdvance: () => void;
}

export function WelcomeSplashStep({ onAdvance }: WelcomeSplashStepProps) {
  return (
    <div className="flex flex-col items-center justify-center text-center max-w-lg mx-auto gap-6">
      <div className="flex flex-col items-center gap-3">
        <h1 className="text-4xl font-semibold tracking-tight text-foreground">
          Real-time generative video.
        </h1>
        <p className="text-xl text-foreground/70">
          For performers and creators.
        </p>
      </div>

      <p className="text-base text-muted-foreground max-w-sm leading-relaxed">
        Transform live video with AI — in real time, on stage or in the studio.
      </p>

      <div className="flex flex-col items-center gap-3 mt-2">
        <button
          onClick={onAdvance}
          className="px-8 py-3 bg-foreground text-background rounded-lg text-base font-medium hover:bg-foreground/90 transition-colors"
        >
          Get Started →
        </button>
        <button
          onClick={onAdvance}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          Skip
        </button>
      </div>
    </div>
  );
}
