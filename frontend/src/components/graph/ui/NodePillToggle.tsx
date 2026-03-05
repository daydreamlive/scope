import { NODE_TOKENS } from "./tokens";

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
    <div
      className={`${NODE_TOKENS.pill} w-[110px] flex items-center justify-center ${className}`}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        disabled={disabled}
        className="w-3 h-3"
      />
    </div>
  );
}
