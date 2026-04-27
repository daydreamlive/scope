import { NODE_TOKENS } from "./tokens";

export interface NodePillSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
  reason?: string | null;
}

interface NodePillSelectProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  options: Array<NodePillSelectOption>;
  className?: string;
}

export function NodePillSelect({
  value,
  onChange,
  disabled = false,
  options,
  className = "",
}: NodePillSelectProps) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      disabled={disabled}
      className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} ${className}`}
    >
      {options.map(opt => (
        <option
          key={opt.value}
          value={opt.value}
          disabled={opt.disabled}
          title={opt.reason ?? undefined}
        >
          {opt.disabled ? `${opt.label} — unavailable` : opt.label}
        </option>
      ))}
    </select>
  );
}
