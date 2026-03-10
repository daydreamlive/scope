import { useState, useRef, useEffect, useCallback } from "react";
import type { ReactNode } from "react";
import { Lock, LockOpen, Pin, PinOff } from "lucide-react";
import { NODE_TOKENS } from "./tokens";
import { useIsZoomedOut } from "../hooks/useIsZoomedOut";
import { useNodeFlags, useNodeFlagToggle } from "../hooks/useNodeFlags";

interface NodeHeaderProps {
  title: string;
  dotColor: string;
  className?: string;
  onTitleChange?: (newTitle: string) => void;
  // Optional right-side content (e.g. play button)
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
      // Focus
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

  const zoomedOut = useIsZoomedOut();
  const { locked, pinned, selected } = useNodeFlags();
  const { toggleLock, togglePin } = useNodeFlagToggle();

  // Icons visible on hover or selection
  const iconsVisible = selected;

  return (
    <div
      className={`${NODE_TOKENS.header} ${rightContent ? "justify-between" : ""} ${className}`}
      style={{ pointerEvents: "auto" }}
    >
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <div
          className={`w-[10px] h-[10px] rounded-full ${dotColor} shrink-0`}
        />
        {zoomedOut ? (
          <div className="h-[10px] flex-1 rounded-full bg-[#fafafa]/10" />
        ) : editing ? (
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

      {!zoomedOut && (
        <div className="flex items-center gap-0.5 shrink-0">
          {/* Lock/Pin icons */}
          <div
            className={`flex items-center gap-0.5 transition-opacity duration-150 ${
              iconsVisible ? "opacity-100" : "opacity-0 group-hover:opacity-100"
            }`}
          >
            <button
              type="button"
              onClick={e => {
                e.stopPropagation();
                toggleLock();
              }}
              className={`p-0.5 rounded transition-colors ${
                locked
                  ? "text-amber-400 hover:text-amber-300"
                  : "text-[#666] hover:text-[#999]"
              }`}
              title={locked ? "Unlock parameters" : "Lock parameters"}
            >
              {locked ? (
                <Lock className="h-3 w-3" />
              ) : (
                <LockOpen className="h-3 w-3" />
              )}
            </button>
            <button
              type="button"
              onClick={e => {
                e.stopPropagation();
                togglePin();
              }}
              className={`p-0.5 rounded transition-colors ${
                pinned
                  ? "text-blue-400 hover:text-blue-300"
                  : "text-[#666] hover:text-[#999]"
              }`}
              title={pinned ? "Unpin node" : "Pin node in place"}
            >
              {pinned ? (
                <Pin className="h-3 w-3" />
              ) : (
                <PinOff className="h-3 w-3" />
              )}
            </button>
          </div>
          {rightContent}
        </div>
      )}
    </div>
  );
}
