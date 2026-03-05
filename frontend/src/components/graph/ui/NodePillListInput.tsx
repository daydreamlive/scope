import { useState, useEffect } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodePillListInputProps {
  value: number[];
  onChange: (value: number[]) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function NodePillListInput({
  value,
  onChange,
  disabled = false,
  placeholder = "e.g. 1000, 750, 500",
  className = "",
}: NodePillListInputProps) {
  const [inputValue, setInputValue] = useState(() => {
    return Array.isArray(value) ? value.join(", ") : "";
  });

  useEffect(() => {
    if (Array.isArray(value)) {
      setInputValue(value.join(", "));
    }
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setInputValue(text);

    const parts = text
      .split(",")
      .map(s => s.trim())
      .filter(s => s);
    const numbers = parts
      .map(s => {
        const num = Number(s);
        return Number.isNaN(num) ? null : num;
      })
      .filter((n): n is number => n !== null);

    if (numbers.length > 0) {
      onChange(numbers);
    } else if (text === "") {
      onChange([]);
    }
  };

  return (
    <input
      type="text"
      value={inputValue}
      onChange={handleChange}
      disabled={disabled}
      placeholder={placeholder}
      className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} ${className}`}
    />
  );
}

