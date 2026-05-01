import { useState } from "react";
import { Cloud, Monitor } from "lucide-react";
import { Button } from "../ui/button";

type InferenceMode = "local" | "cloud";

interface InferenceModeStepProps {
  onSelect: (mode: InferenceMode) => void;
}

const MODES: {
  mode: InferenceMode;
  icon: typeof Cloud;
  title: string;
  description: string;
  detail: string;
}[] = [
  {
    mode: "cloud",
    icon: Cloud,
    title: "Use Daydream Cloud",
    description: "Use cloud GPU provided by Daydream",
    detail: "Requires credits — get started with free credits",
  },
  {
    mode: "local",
    icon: Monitor,
    title: "Run Locally",
    description: "Use a local GPU",
    detail: "Most workflows require at least 24GB of VRAM",
  },
];

export function InferenceModeStep({ onSelect }: InferenceModeStepProps) {
  const [selected, setSelected] = useState<InferenceMode | null>(null);

  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-2xl mx-auto">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-semibold text-foreground">
          Welcome to Daydream Scope
        </h1>
        <p className="text-sm text-muted-foreground/70">
          How would you like to proceed?
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-4 w-full">
        {MODES.map(({ mode, icon: Icon, title, description, detail }) => (
          <button
            key={mode}
            onClick={() => setSelected(mode)}
            className={`flex-1 flex flex-col items-center gap-3 p-6 rounded-xl border-2 transition-all cursor-pointer text-center ${
              selected === mode
                ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                : "border-border bg-card/50 hover:border-border/80 hover:bg-card"
            }`}
          >
            <div
              className={`p-3 rounded-xl transition-colors ${
                selected === mode ? "bg-foreground/10" : "bg-muted"
              }`}
            >
              <Icon className="h-6 w-6 text-foreground" />
            </div>
            <div className="space-y-1">
              <p className="text-base font-medium text-foreground">{title}</p>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
            <p className="text-xs text-muted-foreground/70">{detail}</p>
          </button>
        ))}
      </div>

      <div className="flex flex-col items-center gap-3">
        <Button
          onClick={() => selected && onSelect(selected)}
          disabled={!selected}
          className="px-8"
        >
          Continue
        </Button>
        <p className="text-xs text-muted-foreground">
          You can change this anytime in Settings.
        </p>
      </div>
    </div>
  );
}
