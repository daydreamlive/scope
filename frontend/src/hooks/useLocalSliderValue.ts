import { useState, useEffect, useCallback } from "react";

/**
 * Custom hook for managing local slider state with immediate UI feedback.
 * Syncs with external prop value and provides handlers for SliderWithInput component.
 */
export function useLocalSliderValue(
  value: number,
  onChange?: (value: number) => void,
  decimalPlaces: number = 2
) {
  const [localValue, setLocalValue] = useState<number>(value);

  // Sync with external value changes
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleValueChange = useCallback((newValue: number) => {
    setLocalValue(newValue);
  }, []);

  const handleValueCommit = useCallback(
    (newValue: number) => {
      onChange?.(newValue);
    },
    [onChange]
  );

  const formatValue = useCallback(
    (v: number) => {
      const multiplier = Math.pow(10, decimalPlaces);
      return Math.round(v * multiplier) / multiplier;
    },
    [decimalPlaces]
  );

  return {
    localValue,
    handleValueChange,
    handleValueCommit,
    formatValue,
  };
}
