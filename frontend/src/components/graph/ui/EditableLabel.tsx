import { useState, useRef, useEffect, useCallback } from "react";

interface EditableLabelProps {
  value: string;
  onCommit: (newValue: string) => void;
  /** Which side the label sits on — controls padding direction. */
  side?: "left" | "right";
}

export function EditableLabel({
  value,
  onCommit,
  side = "right",
}: EditableLabelProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = useCallback(() => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) onCommit(trimmed);
    else setDraft(value);
    setEditing(false);
  }, [draft, value, onCommit]);

  const paddingClass = side === "right" ? "pr-1.5" : "pl-1.5";
  const inputMarginClass = side === "right" ? "" : "ml-1.5";

  if (!editing) {
    return (
      <span
        className={`text-[10px] text-[#999] select-none whitespace-nowrap ${paddingClass} cursor-text`}
        onDoubleClick={e => {
          e.stopPropagation();
          setDraft(value);
          setEditing(true);
        }}
        title="Double-click to rename"
      >
        {value}
      </span>
    );
  }

  return (
    <input
      ref={inputRef}
      className={`text-[10px] text-[#ccc] bg-[#333] border border-[#555] rounded px-0.5 outline-none w-16 ${inputMarginClass}`}
      value={draft}
      onChange={e => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          setDraft(value);
          setEditing(false);
        }
      }}
    />
  );
}
