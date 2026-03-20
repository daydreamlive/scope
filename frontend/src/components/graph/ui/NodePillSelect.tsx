import { NODE_TOKENS } from "./tokens";

interface NodePillSelectProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  options: Array<{ value: string; label: string }>;
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
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
