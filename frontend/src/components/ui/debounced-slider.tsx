import { useState, useEffect, useRef } from "react";
import { Slider } from "./slider";

interface DebouncedSliderProps {
  value: number[];
  onValueChange: (value: number[]) => void;
  onValueCommit?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  debounceMs?: number;
}

/**
 * A slider component that debounces value changes.
 * - Updates local state immediately for smooth UI feedback
 * - Calls onValueChange immediately for display updates
 * - Calls onValueCommit with a debounce delay for expensive operations (e.g., API calls)
 */
export function DebouncedSlider({
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className = "",
  debounceMs = 100,
}: DebouncedSliderProps) {
  const commitTimeoutRef = useRef<number | null>(null);
  const [localValue, setLocalValue] = useState<number[]>(value);
  const [isInteracting, setIsInteracting] = useState(false);

  // Sync with external value changes only when not actively interacting
  const displayValue = isInteracting ? localValue : value;

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (commitTimeoutRef.current) {
        clearTimeout(commitTimeoutRef.current);
      }
    };
  }, []);

  const handleValueChange = (newValue: number[]) => {
    setIsInteracting(true);
    setLocalValue(newValue);
    onValueChange(newValue);

    // Debounce the commit callback
    if (onValueCommit) {
      if (commitTimeoutRef.current) {
        clearTimeout(commitTimeoutRef.current);
      }

      commitTimeoutRef.current = setTimeout(() => {
        onValueCommit(newValue);
        setIsInteracting(false);
      }, debounceMs);
    } else {
      // No commit callback, reset interaction state after debounce
      if (commitTimeoutRef.current) {
        clearTimeout(commitTimeoutRef.current);
      }
      commitTimeoutRef.current = setTimeout(() => {
        setIsInteracting(false);
      }, debounceMs);
    }
  };

  return (
    <Slider
      value={displayValue}
      onValueChange={handleValueChange}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      className={className}
    />
  );
}
