import { useState, useRef, useEffect } from "react";
import { NODE_TOKENS } from "./tokens";

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
