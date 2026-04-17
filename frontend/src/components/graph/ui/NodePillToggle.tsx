interface NodePillToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
}

export function NodePillToggle({
  checked,
  onChange,
  disabled = false,
  className = "",
}: NodePillToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`
        relative inline-flex h-[16px] w-[30px] shrink-0 cursor-pointer items-center
        rounded-full border border-[rgba(255,255,255,0.06)] transition-colors duration-200
        focus:outline-none focus:ring-1 focus:ring-blue-400/50
        ${checked ? "bg-blue-500/80" : "bg-[#1b1a1a]"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        ${className}
      `}
    >
      <span
        className={`
          pointer-events-none inline-block h-[12px] w-[12px] rounded-full
          shadow-sm transition-transform duration-200
          ${checked ? "translate-x-[14px] bg-white" : "translate-x-[1px] bg-[#666]"}
        `}
      />
    </button>
  );
}
