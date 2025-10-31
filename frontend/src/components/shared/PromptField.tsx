import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";
import { X } from "lucide-react";
import type { PromptItem } from "../../lib/api";

interface PromptFieldProps {
  prompt: PromptItem;
  index: number;
  placeholder: string;
  showRemove: boolean;
  focusedIndex: number | null;
  onTextChange: (index: number, text: string) => void;
  onFocus: (index: number) => void;
  onBlur: () => void;
  onRemove: (index: number) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
  disabled?: boolean;
}

export function PromptField({
  prompt,
  index,
  placeholder,
  showRemove,
  focusedIndex,
  onTextChange,
  onFocus,
  onBlur,
  onRemove,
  onKeyDown,
  disabled = false,
}: PromptFieldProps) {
  const isFocused = focusedIndex === index;

  return (
    <>
      <Textarea
        placeholder={placeholder}
        value={prompt.text}
        onChange={e => onTextChange(index, e.target.value)}
        onKeyDown={onKeyDown}
        onFocus={() => onFocus(index)}
        onBlur={onBlur}
        disabled={disabled}
        rows={isFocused ? 3 : 1}
        className={`flex-1 resize-none bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed ${
          isFocused
            ? "min-h-[80px]"
            : "min-h-[24px] overflow-hidden whitespace-nowrap text-ellipsis"
        }`}
      />
      {showRemove && (
        <Button
          onClick={() => onRemove(index)}
          disabled={disabled}
          size="sm"
          variant="ghost"
          className="rounded-full w-8 h-8 p-0"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </>
  );
}
