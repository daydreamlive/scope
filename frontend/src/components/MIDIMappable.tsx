import { useEffect, useState, useMemo } from "react";
import { useMIDI } from "../contexts/MIDIContext";
import { cn } from "../lib/utils";

interface MIDIMappableProps {
  children: React.ReactNode;
  parameterId?: string;
  arrayIndex?: number;
  actionId?: string;
  mappingType?: "continuous" | "toggle" | "trigger" | "enum_cycle";
  range?: { min: number; max: number };
  enumValues?: string[];
  className?: string;
  disabled?: boolean;
}

export function MIDIMappable({
  children,
  parameterId,
  arrayIndex,
  actionId,
  mappingType,
  range,
  enumValues,
  className,
  disabled = false,
}: MIDIMappableProps) {
  const {
    isMappingMode,
    learningParameter,
    startLearning,
    cancelLearning,
    getMappedSource,
  } = useMIDI();

  const [justMapped, setJustMapped] = useState(false);

  const paramId = useMemo(() => {
    if (actionId) return actionId;
    if (arrayIndex !== undefined) return `${parameterId}[${arrayIndex}]`;
    return parameterId || "";
  }, [parameterId, arrayIndex, actionId]);

  const isLearning = learningParameter === paramId;

  const mappedSource = parameterId || actionId
    ? getMappedSource(parameterId || "", arrayIndex, actionId)
    : null;

  const isMapped = mappedSource !== null;

  const handleClick = (e: React.MouseEvent) => {
    if (disabled) return;
    if (isMappingMode && !isLearning) {
      e.preventDefault();
      e.stopPropagation();
      if (parameterId || actionId) {
        startLearning(parameterId || "", arrayIndex, actionId, mappingType, range, enumValues);
      }
    }
  };

  useEffect(() => {
    if (isMapped && !isLearning && learningParameter === null) {
      setJustMapped(true);
      const timer = setTimeout(() => setJustMapped(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [isMapped, isLearning, learningParameter]);

  useEffect(() => {
    return () => { if (isLearning) cancelLearning(); };
  }, [isLearning, cancelLearning]);

  if (!parameterId && !actionId) return <>{children}</>;

  return (
    <div
      className={cn("relative", isMappingMode && !disabled && "cursor-pointer", className)}
      onClick={handleClick}
    >
      {isMappingMode && !disabled && (
        <div
          className={cn(
            "absolute inset-0 z-10 rounded-md border-2 transition-all",
            isLearning
              ? "border-blue-500 bg-blue-500/10 animate-pulse"
              : "border-blue-400/50 bg-blue-400/5 hover:border-blue-400 hover:bg-blue-400/10"
          )}
          style={{ pointerEvents: "auto" }}
        >
          {isLearning && (
            <div className="absolute inset-0 flex items-center justify-center bg-blue-500/20 rounded-md">
              <span className="text-xs font-medium text-blue-700 dark:text-blue-300">
                Waiting for MIDI...
              </span>
            </div>
          )}
        </div>
      )}

      {justMapped && (
        <div className="absolute inset-0 z-20 rounded-md border-2 border-green-500 bg-green-500/20 animate-pulse" />
      )}

      {isMapped && !isMappingMode && (
        <div
          className="absolute -top-0.5 -right-0.5 z-10 w-1.5 h-1.5 rounded-full bg-blue-500"
          title={`Mapped to ${mappedSource}`}
        />
      )}

      <div className={cn(isMappingMode && !disabled && "pointer-events-none")}>
        {children}
      </div>
    </div>
  );
}
