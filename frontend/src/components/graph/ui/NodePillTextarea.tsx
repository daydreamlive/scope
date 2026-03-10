import { useRef } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodePillTextareaProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: () => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function NodePillTextarea({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder,
  className = "",
}: NodePillTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit?.();
      textareaRef.current?.blur();
    }
  };

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      onWheel={handleWheel}
      disabled={disabled}
      placeholder={placeholder}
      rows={3}
      className={`${NODE_TOKENS.pillInput} !rounded-md w-full min-w-[110px] resize-y min-h-[60px] text-left py-1.5 leading-relaxed nowheel ${className}`}
    />
  );
}
