import { Play, Pause } from "lucide-react";

interface PlayOverlayProps {
  isPlaying?: boolean;
  onClick?: () => void;
  size?: "sm" | "md" | "lg";
  className?: string;
  variant?: "default" | "themed";
}

const sizeClasses = {
  sm: {
    circle: "w-16 h-16",
    icon: "w-6 h-6",
    padding: "p-3",
  },
  md: {
    circle: "w-20 h-20",
    icon: "w-8 h-8",
    padding: "p-3",
  },
  lg: {
    circle: "w-24 h-24",
    icon: "w-12 h-12",
    padding: "p-4",
  },
};

export function PlayOverlay({
  isPlaying = false,
  onClick,
  size = "lg",
  className = "",
  variant = "default",
}: PlayOverlayProps) {
  const sizes = sizeClasses[size];

  if (variant === "themed") {
    return (
      <div
        className={`${sizes.circle} rounded-full border-2 border-input bg-background hover:bg-accent transition-colors flex items-center justify-center cursor-pointer shadow-lg ${className}`}
        onClick={onClick}
      >
        {isPlaying ? (
          <Pause className={`${sizes.icon} text-foreground`} />
        ) : (
          <Play className={`${sizes.icon} text-foreground ml-0.5`} />
        )}
      </div>
    );
  }

  // Default variant - semi-transparent black background with white icons
  return (
    <div
      className={`bg-black/50 rounded-full ${sizes.padding} transition-colors hover:bg-black/60 cursor-pointer ${className}`}
      onClick={onClick}
    >
      {isPlaying ? (
        <Pause className={`${sizes.icon} text-white`} />
      ) : (
        <Play className={`${sizes.icon} text-white`} />
      )}
    </div>
  );
}
