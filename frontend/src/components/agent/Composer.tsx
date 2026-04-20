import { useRef, useState, type KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ComposerProps {
  onSend: (text: string) => Promise<void>;
  disabled?: boolean;
  placeholder?: string;
}

export function Composer({ onSend, disabled, placeholder }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const send = async () => {
    const text = value.trim();
    if (!text || disabled) return;
    setValue("");
    await onSend(text);
    textareaRef.current?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl+Enter or bare Enter to send (Shift+Enter inserts newline).
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <div className="border-t border-[rgba(255,255,255,0.08)] p-3 flex items-end gap-2">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        rows={2}
        placeholder={
          placeholder ?? (disabled ? "Agent is working…" : "Ask the agent…")
        }
        className="flex-1 resize-none rounded-md bg-[#141414] border border-[rgba(255,255,255,0.08)] px-3 py-2 text-sm text-[#e6e6e6] placeholder:text-[#595959] focus:outline-none focus:border-[#1f6feb] disabled:opacity-60"
        disabled={disabled}
      />
      <Button
        size="sm"
        onClick={send}
        disabled={disabled || !value.trim()}
        className="gap-1.5"
      >
        <Send className="h-3.5 w-3.5" />
        Send
      </Button>
    </div>
  );
}
