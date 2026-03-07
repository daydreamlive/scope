import { useState, useRef, useEffect, useCallback } from "react";
import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodeHeaderProps {
  title: string;
  dotColor: string;
  className?: string;
  onTitleChange?: (newTitle: string) => void;
  /** Optional content rendered on the right side of the header (e.g. a play button). */
  rightContent?: ReactNode;
}

export function NodeHeader({
  title,
  dotColor,
  className = "",
  onTitleChange,
  rightContent,
}: NodeHeaderProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      setDraft(title);
      // Focus after mount
      requestAnimationFrame(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      });
    }
  }, [editing, title]);

  const commit = useCallback(() => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== title && onTitleChange) {
      onTitleChange(trimmed);
    }
  }, [draft, title, onTitleChange]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        setEditing(false);
      }
    },
    [commit]
  );

  const handleDoubleClick = useCallback(() => {
    if (onTitleChange) {
      setEditing(true);
    }
  }, [onTitleChange]);

  return (
    <div
      className={`${NODE_TOKENS.header} ${rightContent ? "justify-between" : ""} ${className}`}
    >
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <div
          className={`w-[10px] h-[10px] rounded-full ${dotColor} shrink-0`}
        />
        {editing ? (
          <input
            ref={inputRef}
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={handleKeyDown}
            className={`${NODE_TOKENS.headerText} bg-transparent border-none outline-none p-0 m-0 w-full`}
            spellCheck={false}
          />
        ) : (
          <p
            className={`${NODE_TOKENS.headerText} truncate`}
            onDoubleClick={handleDoubleClick}
          >
            {title}
          </p>
        )}
      </div>
      {rightContent}
    </div>
  );
}
