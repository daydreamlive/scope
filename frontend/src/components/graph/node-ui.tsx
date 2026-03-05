import {
  type ReactNode,
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  useCallback,
} from "react";
import { NodeResizer } from "@xyflow/react";

export const NODE_TOKENS = {
  card: "bg-[#2a2a2a] border border-[rgba(119,119,119,0.55)] rounded-xl min-w-[240px] relative w-full h-full flex flex-col",
  cardSelected: "ring-2 ring-blue-400/50",
  header:
    "bg-[#181717] border-b border-[rgba(119,119,119,0.15)] flex items-center gap-2 px-2 py-1 h-[28px] rounded-t-xl",
  body: "py-1.5 px-4",
  bodyWithGap: "py-1.5 px-4 flex flex-col gap-1.5",
  pill: "bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5",
  pillInput:
    "bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5 text-[#fafafa] text-[10px] appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/50 w-[110px]",
  pillInputText: "text-center",
  pillInputNumber:
    "text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none",
  labelText: "text-[#8c8c8d] text-[10px] font-normal",
  primaryText: "text-[#fafafa] text-[10px] font-normal",
  headerText: "text-[#fafafa] text-[12px] font-normal",
  paramRow: "flex items-center justify-between h-[20px]",
  sectionTitle: "text-[10px] font-normal text-[#8c8c8d] mb-2",
  panelBackground: "bg-[#181717]",
  panelBorder: "border-[rgba(119,119,119,0.15)]",
  toolbar:
    "flex items-center gap-2 px-4 py-2 bg-[#181717] border-b border-[rgba(119,119,119,0.15)]",
  toolbarButton:
    "px-3 py-1.5 text-xs font-medium rounded-lg bg-[#2a2a2a] border border-[rgba(119,119,119,0.35)] text-[#fafafa] hover:bg-[#2a2a2a]/80 transition-colors",
  toolbarStatus: "text-xs text-[#8c8c8d] ml-2",
} as const;

interface NodeCardProps {
  children: ReactNode;
  selected?: boolean;
  className?: string;
  /** When true, measures content height and enforces it as minHeight on resize */
  autoMinHeight?: boolean;
}

export function NodeCard({
  children,
  selected,
  className = "",
  autoMinHeight = false,
}: NodeCardProps) {
  const measureRef = useRef<HTMLDivElement>(null);
  const [minH, setMinH] = useState(60);

  useLayoutEffect(() => {
    if (!autoMinHeight || !measureRef.current) return;
    const h = measureRef.current.offsetHeight;
    setMinH(prev => (Math.abs(h - prev) > 1 ? h : prev));
  });

  return (
    <div
      className={`${NODE_TOKENS.card} ${selected ? NODE_TOKENS.cardSelected : ""} ${className}`}
    >
      <NodeResizer
        isVisible={!!selected}
        minWidth={240}
        minHeight={autoMinHeight ? Math.max(60, minH) : 60}
        lineClassName="!border-transparent"
        handleClassName="!w-2 !h-2 !bg-transparent !border !border-blue-400/20 hover:!border-blue-400/40 !rounded-sm"
      />
      {autoMinHeight ? (
        <div ref={measureRef} className="flex flex-col w-full">
          {children}
        </div>
      ) : (
        children
      )}
    </div>
  );
}

interface NodeHeaderProps {
  title: string;
  dotColor: string;
  className?: string;
}

export function NodeHeader({
  title,
  dotColor,
  className = "",
}: NodeHeaderProps) {
  return (
    <div className={`${NODE_TOKENS.header} ${className}`}>
      <div className={`w-[10px] h-[10px] rounded-full ${dotColor} shrink-0`} />
      <p className={NODE_TOKENS.headerText}>{title}</p>
    </div>
  );
}

interface NodeBodyProps {
  children: ReactNode;
  withGap?: boolean;
  className?: string;
}

export function NodeBody({
  children,
  withGap = false,
  className = "",
}: NodeBodyProps) {
  return (
    <div
      className={`${withGap ? NODE_TOKENS.bodyWithGap : NODE_TOKENS.body} flex-1 min-h-0 ${className}`}
    >
      {children}
    </div>
  );
}

interface NodeSectionProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export function NodeSection({
  title,
  children,
  className = "",
}: NodeSectionProps) {
  return (
    <div className={className}>
      <h4 className={NODE_TOKENS.sectionTitle}>{title}</h4>
      <div className="flex flex-col gap-3">{children}</div>
    </div>
  );
}

interface NodeParamRowProps {
  label: string;
  children: ReactNode;
  className?: string;
}

export function NodeParamRow({
  label,
  children,
  className = "",
}: NodeParamRowProps) {
  return (
    <div className={`${NODE_TOKENS.paramRow} ${className}`}>
      <p className={`${NODE_TOKENS.labelText} w-[80px] shrink-0 truncate`}>
        {label}
      </p>
      {children}
    </div>
  );
}

interface NodePillProps {
  children: ReactNode;
  className?: string;
}

export function NodePill({ children, className = "" }: NodePillProps) {
  return (
    <div
      className={`${NODE_TOKENS.pill} w-[110px] flex items-center justify-center ${className}`}
    >
      <p className={`${NODE_TOKENS.primaryText} leading-[1.55]`}>{children}</p>
    </div>
  );
}

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

interface NodePillSearchableSelectProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  options: Array<{ value: string; label: string }>;
  className?: string;
  placeholder?: string;
}

export function NodePillSearchableSelect({
  value,
  onChange,
  disabled = false,
  options,
  className = "",
  placeholder = "Search...",
}: NodePillSearchableSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollableRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
        setSearchText("");
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      setTimeout(() => inputRef.current?.focus(), 0);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  // Prevent graph zoom when scrolling dropdown
  useEffect(() => {
    const scrollable = scrollableRef.current;
    if (!scrollable) return;

    const handleWheel = (e: WheelEvent) => {
      e.stopPropagation();
      e.preventDefault();
      scrollable.scrollTop += e.deltaY;
    };

    scrollable.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      scrollable.removeEventListener("wheel", handleWheel);
    };
  }, [isOpen]);

  const filteredOptions = options.filter(
    opt =>
      opt.label.toLowerCase().includes(searchText.toLowerCase()) ||
      opt.value.toLowerCase().includes(searchText.toLowerCase())
  );

  const selectedOption = options.find(opt => opt.value === value);
  const displayText = selectedOption?.label || placeholder;

  const handleSelect = (optionValue: string) => {
    onChange(optionValue);
    setIsOpen(false);
    setSearchText("");
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} w-[110px] text-left cursor-pointer flex items-center justify-between`}
      >
        <span className="truncate">{displayText}</span>
        <span className="ml-1 shrink-0">▼</span>
      </button>

      {isOpen && (
        <div
          className="absolute z-50 mt-1 w-[200px] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-lg shadow-lg max-h-[240px] overflow-hidden flex flex-col nowheel"
          onMouseDown={e => e.stopPropagation()}
          onWheel={e => e.stopPropagation()}
        >
          <input
            ref={inputRef}
            type="text"
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            placeholder={placeholder}
            className="px-2 py-1 text-[10px] bg-[#2a2a2a] border-b border-[rgba(119,119,119,0.15)] text-[#fafafa] focus:outline-none focus:ring-1 focus:ring-blue-400/50"
            onMouseDown={e => e.stopPropagation()}
            onWheel={e => e.stopPropagation()}
          />
          <div
            ref={scrollableRef}
            className="overflow-y-auto overflow-x-hidden max-h-[200px] nowheel [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-black/50 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-black/70"
            style={{
              scrollbarWidth: "thin",
              scrollbarColor: "rgba(0,0,0,0.5) transparent",
            }}
          >
            {filteredOptions.length === 0 ? (
              <div className="px-2 py-1 text-[10px] text-[#8c8c8d] text-center">
                No matches
              </div>
            ) : (
              filteredOptions.map(opt => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => handleSelect(opt.value)}
                  className={`w-full px-2 py-1 text-[10px] text-left hover:bg-[#2a2a2a] transition-colors truncate ${
                    opt.value === value
                      ? "bg-[#2a2a2a] text-blue-400"
                      : "text-[#fafafa]"
                  }`}
                  onMouseDown={e => e.stopPropagation()}
                >
                  {opt.label}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface NodePillInputProps {
  type: "text" | "number";
  value: string | number;
  onChange: (value: string | number) => void;
  disabled?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  className?: string;
}

export function NodePillInput({
  type,
  value,
  onChange,
  disabled = false,
  placeholder,
  min,
  max,
  className = "",
}: NodePillInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<{
    startX: number;
    startValue: number;
    hasDragged: boolean;
  } | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (type === "number") {
      const numValue = Number(e.target.value);
      if (!Number.isNaN(numValue)) {
        onChange(numValue);
      }
    } else {
      onChange(e.target.value);
    }
  };

  const clampValue = useCallback(
    (v: number) => {
      let clamped = v;
      if (min !== undefined) clamped = Math.max(min, clamped);
      if (max !== undefined) clamped = Math.min(max, clamped);
      return clamped;
    },
    [min, max]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (type !== "number" || disabled) return;
      if (document.activeElement === inputRef.current) return;

      e.preventDefault();
      e.stopPropagation();
      dragRef.current = {
        startX: e.clientX,
        startValue: Number(value) || 0,
        hasDragged: false,
      };

      const sensitivity =
        min !== undefined && max !== undefined ? (max - min) / 300 : 0.5;

      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const dx = ev.clientX - dragRef.current.startX;
        if (!dragRef.current.hasDragged && Math.abs(dx) < 3) return;
        dragRef.current.hasDragged = true;
        const newVal = clampValue(
          dragRef.current.startValue + dx * sensitivity
        );
        onChange(
          sensitivity >= 1
            ? Math.round(newVal)
            : Math.round(newVal * 1000) / 1000
        );
      };

      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (!dragRef.current?.hasDragged) {
          inputRef.current?.focus();
          inputRef.current?.select();
        }
        dragRef.current = null;
      };

      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [type, disabled, value, min, max, clampValue, onChange]
  );

  const isNumber = type === "number";

  return (
    <input
      ref={inputRef}
      type={type}
      value={value}
      onChange={handleChange}
      onMouseDown={isNumber ? handleMouseDown : undefined}
      disabled={disabled}
      placeholder={placeholder}
      min={min}
      max={max}
      className={`${NODE_TOKENS.pillInput} ${isNumber ? NODE_TOKENS.pillInputNumber : NODE_TOKENS.pillInputText} ${isNumber && !disabled ? "cursor-ew-resize focus:cursor-text" : ""} ${isNumber ? "nodrag" : ""} ${className}`}
    />
  );
}

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

interface NodePillTextareaProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function NodePillTextarea({
  value,
  onChange,
  disabled = false,
  placeholder,
  className = "",
}: NodePillTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
  };

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      onWheel={handleWheel}
      disabled={disabled}
      placeholder={placeholder}
      rows={3}
      className={`${NODE_TOKENS.pillInput} !rounded-md w-full min-w-[110px] resize-y min-h-[60px] text-left py-1.5 leading-relaxed nowheel ${className}`}
    />
  );
}

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
