import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
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
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollableRef = useRef<HTMLDivElement>(null);
  const [dropdownPos, setDropdownPos] = useState<{
    top: number;
    left: number;
  }>({ top: 0, left: 0 });

  const updatePosition = useCallback(() => {
    if (!buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    setDropdownPos({ top: rect.bottom + 4, left: rect.left });
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    updatePosition();
    setTimeout(() => inputRef.current?.focus(), 0);

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        buttonRef.current?.contains(target) ||
        dropdownRef.current?.contains(target)
      )
        return;
      setIsOpen(false);
      setSearchText("");
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, updatePosition]);

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
    <div className={`relative ${className}`}>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} w-[110px] text-left cursor-pointer flex items-center justify-between`}
      >
        <span className="truncate">{displayText}</span>
        <span className="ml-1 shrink-0">▼</span>
      </button>

      {isOpen &&
        createPortal(
          <div
            ref={dropdownRef}
            className="fixed z-[9999] w-[200px] bg-[#1a1a1a] border border-[rgba(255,255,255,0.06)] rounded-md shadow-[0_4px_16px_rgba(0,0,0,0.4)] max-h-[240px] overflow-hidden flex flex-col nowheel"
            style={{ top: dropdownPos.top, left: dropdownPos.left }}
            onMouseDown={e => e.stopPropagation()}
            onWheel={e => e.stopPropagation()}
          >
            <input
              ref={inputRef}
              type="text"
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              placeholder={placeholder}
              className="px-2 py-1 text-[10px] bg-[#242424] border-b border-[rgba(255,255,255,0.06)] text-[#fafafa] focus:outline-none focus:ring-1 focus:ring-blue-400/60"
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
          </div>,
          document.body
        )}
    </div>
  );
}
