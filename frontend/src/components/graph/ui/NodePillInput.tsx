import { useRef, useCallback } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodePillInputProps {
  type: "text" | "number";
  value: string | number;
  onChange: (value: string | number) => void;
  onSubmit?: () => void;
  disabled?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}

export function NodePillInput({
  type,
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder,
  min,
  max,
  step,
  className = "",
}: NodePillInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<{
    startX: number;
    startValue: number;
    hasDragged: boolean;
  } | null>(null);

  const isInteger = step !== undefined && step >= 1;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (type === "number") {
      const numValue = Number(e.target.value);
      if (!Number.isNaN(numValue)) {
        onChange(isInteger ? Math.round(numValue) : numValue);
      }
    } else {
      onChange(e.target.value);
    }
  };

  const clampValue = useCallback(
    (v: number) => {
      let clamped = v;
      if (min !== undefined) clamped = Math.max(min, clamped);
      if (max !== undefined) clamped = Math.min(max, clamped);
      return clamped;
    },
    [min, max]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        onSubmit?.();
        inputRef.current?.blur();
      }
    },
    [onSubmit]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (type !== "number" || disabled) return;
      if (document.activeElement === inputRef.current) return;

      e.preventDefault();
      e.stopPropagation();
      dragRef.current = {
        startX: e.clientX,
        startValue: Number(value) || 0,
        hasDragged: false,
      };

      const sensitivity =
        min !== undefined && max !== undefined ? (max - min) / 300 : 0.5;

      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const dx = ev.clientX - dragRef.current.startX;
        if (!dragRef.current.hasDragged && Math.abs(dx) < 3) return;
        dragRef.current.hasDragged = true;
        const newVal = clampValue(
          dragRef.current.startValue + dx * sensitivity
        );
        onChange(
          isInteger ? Math.round(newVal) : Math.round(newVal * 1000) / 1000
        );
      };

      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (!dragRef.current?.hasDragged) {
          inputRef.current?.focus();
          inputRef.current?.select();
        }
        dragRef.current = null;
      };

      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [type, disabled, value, min, max, clampValue, onChange, isInteger]
  );

  const isNumber = type === "number";

  return (
    <input
      ref={inputRef}
      type={type}
      value={value}
      onChange={handleChange}
      onKeyDown={onSubmit ? handleKeyDown : undefined}
      onMouseDown={isNumber ? handleMouseDown : undefined}
      disabled={disabled}
      placeholder={placeholder}
      min={min}
      max={max}
      step={step}
      className={`${NODE_TOKENS.pillInput} ${isNumber ? NODE_TOKENS.pillInputNumber : NODE_TOKENS.pillInputText} ${isNumber && !disabled ? "cursor-ew-resize focus:cursor-text" : ""} ${isNumber ? "nodrag" : ""} ${className}`}
    />
  );
}
