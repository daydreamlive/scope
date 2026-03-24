import { useCallback, useRef } from "react";

interface UseSliderOptions {
  min: number;
  max: number;
  step: number;
  value: number;
  /** Called with the new clamped+stepped value. */
  onChange: (value: number) => void;
  /** Number of decimal places for toFixed (default: 10). */
  precision?: number;
}

/**
 * Shared drag-slider logic used by SliderNode, VaceNode, and LoraNode.
 *
 * Returns a ref to attach to the slider track element, the clamped value,
 * a percentage for the fill, and a pointerDown handler.
 */
export function useSlider({
  min,
  max,
  step,
  value,
  onChange,
  precision = 10,
}: UseSliderOptions) {
  const sliderRef = useRef<HTMLDivElement>(null);

  const clampedValue = Math.min(Math.max(value, min), max);
  const pct = max > min ? ((clampedValue - min) / (max - min)) * 100 : 0;

  const setValueFromMouse = useCallback(
    (clientX: number) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.min(Math.max(ratio, 0), 1);
      let newVal = min + ratio * (max - min);
      newVal = min + Math.round((newVal - min) / step) * step;
      newVal = Math.min(Math.max(newVal, min), max);
      onChange(parseFloat(newVal.toFixed(precision)));
    },
    [min, max, step, onChange, precision]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);
      setValueFromMouse(e.clientX);

      const onMove = (ev: PointerEvent) => setValueFromMouse(ev.clientX);
      const onUp = () => {
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [setValueFromMouse]
  );

  return { sliderRef, clampedValue, pct, handlePointerDown };
}
