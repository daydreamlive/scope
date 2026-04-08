import { GraduationCap, Zap } from "lucide-react";

interface OnboardingStyleStepProps {
  onSelect: (style: "teaching" | "simple") => void;
}

const STYLES: {
  style: "teaching" | "simple";
  icon: typeof GraduationCap;
  label: string;
  description: string;
}[] = [
  {
    style: "teaching",
    icon: GraduationCap,
    label: "Guided",
    description:
      "I'm new to real-time AI or node-based tools. Show me how it works.",
  },
  {
    style: "simple",
    icon: Zap,
    label: "Jump In",
    description: "I know my way around. Just get me generating.",
  },
];

export function OnboardingStyleStep({ onSelect }: OnboardingStyleStepProps) {
  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-2xl mx-auto">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-semibold text-foreground">
          How do you want to start?
        </h1>
        <p className="text-sm text-muted-foreground/70">
          Pick a style — you can change this anytime in Settings.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-4 w-full">
        {STYLES.map(({ style, icon: Icon, label, description }) => (
          <button
            key={style}
            onClick={() => onSelect(style)}
            className="flex-1 flex flex-col items-center gap-3 p-6 rounded-xl border-2 border-border bg-card/50 hover:border-border/80 hover:bg-card transition-all cursor-pointer text-center"
          >
            <div className="p-3 rounded-xl bg-muted transition-colors">
              <Icon className="h-6 w-6 text-foreground" />
            </div>
            <div className="space-y-1">
              <p className="text-base font-medium text-foreground">{label}</p>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
