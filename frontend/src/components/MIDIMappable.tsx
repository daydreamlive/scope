import { useEffect, useRef, useState, useMemo } from "react";
import { toast } from "sonner";
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
  mappingModeClassName?: string;
  overlayClassName?: string;
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
  mappingModeClassName,
  overlayClassName,
  disabled = false,
}: MIDIMappableProps) {
  const {
    isMappingMode,
    setMappingMode,
    learningParameter,
    startLearning,
    cancelLearning,
    getMappedSource,
    activeParameters,
  } = useMIDI();

  const [justMapped, setJustMapped] = useState(false);

  const paramId = useMemo(() => {
    if (actionId) return actionId;
    if (arrayIndex !== undefined) return `${parameterId}[${arrayIndex}]`;
    return parameterId || "";
  }, [parameterId, arrayIndex, actionId]);

  const isLearning = learningParameter === paramId;

  const mappedSource =
    parameterId || actionId
      ? getMappedSource(parameterId || "", arrayIndex, actionId)
      : null;

  const isMapped = mappedSource !== null;
  const isActive = activeParameters.has(paramId);

  const handleClick = (e: React.MouseEvent) => {
    if (disabled) return;
    if (isMappingMode && !isLearning) {
      e.preventDefault();
      e.stopPropagation();
      if (parameterId || actionId) {
        startLearning(
          parameterId || "",
          arrayIndex,
          actionId,
          mappingType,
          range,
          enumValues
        );
      }
    }
  };

  const wasLearningRef = useRef(false);
  useEffect(() => {
    if (isLearning) {
      wasLearningRef.current = true;
    } else if (wasLearningRef.current && isMapped) {
      wasLearningRef.current = false;
      setJustMapped(true);
      const timer = setTimeout(() => setJustMapped(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [isMapped, isLearning]);

  useEffect(() => {
    if (isLearning) {
      const id = toast("Listening for MIDI input", {
        icon: (
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
            <span className="h-1.5 w-1.5 rounded-full bg-blue-400/80 animate-pulse [animation-delay:150ms]" />
          </span>
        ),
        duration: Infinity,
      });
      return () => { toast.dismiss(id); };
    }
  }, [isLearning]);

  useEffect(() => {
    return () => {
      if (isLearning) cancelLearning();
    };
  }, [isLearning, cancelLearning]);

  if (!parameterId && !actionId) return <>{children}</>;

  return (
    <div
      className={cn(
        "relative",
        isMappingMode && !disabled && "cursor-pointer p-1.5",
        isMappingMode && !disabled && mappingModeClassName,
        className
      )}
      onClick={handleClick}
    >
      {isMappingMode && !disabled && (
        <div
          className={cn(
            "absolute inset-0 z-10 rounded-md border-2 transition-all",
            overlayClassName,
            isLearning
              ? "border-blue-500 bg-blue-500/10 animate-pulse"
              : isMapped && isActive
                ? "border-green-400 bg-green-400/20"
                : isMapped
                  ? "border-green-400/50 bg-green-400/5 hover:border-green-400 hover:bg-green-400/10"
                  : "border-blue-400/50 bg-blue-400/5 hover:border-blue-400 hover:bg-blue-400/10"
          )}
          style={{ pointerEvents: "auto" }}
        />
      )}

      {justMapped && (
        <div
          className={cn(
            "absolute inset-0 z-20 rounded-md border-2 border-green-500 bg-green-500/20 animate-pulse",
            overlayClassName
          )}
        />
      )}

      {isMapped && !isMappingMode && (
        <>
          {isActive && (
            <div className="absolute top-0 right-0 z-10 w-2 h-2 rounded-full bg-green-500/60 animate-ping" />
          )}
          <div
            className={cn(
              "absolute top-0 right-0 z-10 w-2 h-2 rounded-full transition-colors duration-150 cursor-pointer",
              isActive ? "bg-green-500" : "bg-blue-500"
            )}
            title={`Mapped to ${mappedSource}`}
            onClick={e => {
              e.stopPropagation();
              setMappingMode(true);
            }}
          />
        </>
      )}

      <div className={cn(isMappingMode && !disabled && "pointer-events-none")}>
        {children}
      </div>
    </div>
  );
}
