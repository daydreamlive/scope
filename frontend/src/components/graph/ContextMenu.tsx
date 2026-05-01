import {
  useEffect,
  useRef,
  useState,
  useMemo,
  useLayoutEffect,
  type ReactNode,
} from "react";

export interface ContextMenuItem {
  label: string;
  onClick?: () => void;
  danger?: boolean;
  icon?: ReactNode;
  children?: ContextMenuItem[];
  keywords?: string[];
}

interface ContextMenuProps {
  x: number;
  y: number;
  onClose: () => void;
  items: ContextMenuItem[];
  header?: string;
}

type FlatItem = ContextMenuItem & { parentLabel?: string };

const VIEWPORT_PADDING = 8;

export function ContextMenu({
  x,
  y,
  onClose,
  items,
  header,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [search, setSearch] = useState("");
  const [activeSubmenu, setActiveSubmenu] = useState<number | null>(null);

  const [pos, setPos] = useState<{ left: number; top: number }>({
    left: x,
    top: y,
  });
  const [flipX, setFlipX] = useState(false);
  const [flipY, setFlipY] = useState(false);

  // Measure after first paint and clamp to viewport
  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = x;
    let top = y;
    let fx = false;
    let fy = false;

    if (x + rect.width > vw - VIEWPORT_PADDING) {
      left = x - rect.width;
      fx = true;
    }
    if (left < VIEWPORT_PADDING) left = VIEWPORT_PADDING;

    if (y + rect.height > vh - VIEWPORT_PADDING) {
      top = y - rect.height;
      fy = true;
    }
    if (top < VIEWPORT_PADDING) top = VIEWPORT_PADDING;

    setPos({ left, top });
    setFlipX(fx);
    setFlipY(fy);
  }, [x, y]);

  useEffect(() => {
    setTimeout(() => searchRef.current?.focus(), 50);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    const timeout = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("contextmenu", handleClickOutside);
      document.addEventListener("keydown", handleEscape);
    }, 0);
    return () => {
      clearTimeout(timeout);
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("contextmenu", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  // Flatten hierarchy so children are searchable with their parent label as context
  const flatItems = useMemo<FlatItem[]>(() => {
    const result: FlatItem[] = [];
    for (const item of items) {
      if (item.children && item.children.length > 0) {
        for (const child of item.children) {
          result.push({
            ...child,
            parentLabel: item.label,
            keywords: [...(child.keywords ?? []), item.label],
          });
        }
      } else {
        result.push(item);
      }
    }
    return result;
  }, [items]);

  const searchResults = useMemo<FlatItem[] | null>(() => {
    if (!search.trim()) return null;
    const q = search.toLowerCase();
    return flatItems.filter(
      item =>
        item.label.toLowerCase().includes(q) ||
        item.keywords?.some(k => k.toLowerCase().includes(q))
    );
  }, [search, flatItems]);

  const handleSelect = (item: ContextMenuItem) => {
    item.onClick?.();
    onClose();
  };

  return (
    <div
      ref={menuRef}
      className="fixed z-50 min-w-[180px] rounded-lg bg-[#141414] border border-[rgba(255,255,255,0.06)] shadow-[0_4px_24px_rgba(0,0,0,0.5)] overflow-visible"
      style={{ left: `${pos.left}px`, top: `${pos.top}px` }}
    >
      {/* Search */}
      <div className="px-2 pt-2 pb-1.5">
        <input
          ref={searchRef}
          type="text"
          value={search}
          onChange={e => {
            setSearch(e.target.value);
            setActiveSubmenu(null);
          }}
          placeholder="Search…"
          className="w-full px-2 py-1 text-[12px] bg-[#1a1a1a] border border-[rgba(255,255,255,0.06)] rounded-md text-[#e0e0e0] placeholder-[#444] focus:outline-none focus:border-[rgba(255,255,255,0.18)]"
          onMouseDown={e => e.stopPropagation()}
        />
      </div>

      {header && searchResults === null && (
        <div className="px-3 pt-0.5 pb-1 text-[10px] font-semibold tracking-widest text-[#444] uppercase select-none">
          {header}
        </div>
      )}

      <div className="pb-1.5">
        {searchResults !== null ? (
          searchResults.length === 0 ? (
            <div className="px-3 py-2 text-[12px] text-[#444] text-center select-none">
              No results
            </div>
          ) : (
            searchResults.map((item, i) => (
              <button
                key={i}
                onClick={() => handleSelect(item)}
                className={`w-full text-left px-3 py-1.5 text-[13px] flex items-center gap-2.5 transition-colors ${
                  item.danger
                    ? "text-red-400 hover:bg-[rgba(255,255,255,0.05)] hover:text-red-300"
                    : "text-[#e0e0e0] hover:bg-[rgba(255,255,255,0.05)]"
                }`}
              >
                {item.icon && (
                  <span className="text-[#777] shrink-0 [&_svg]:w-[15px] [&_svg]:h-[15px]">
                    {item.icon}
                  </span>
                )}
                <span>{item.label}</span>
                {item.parentLabel && (
                  <span className="ml-auto text-[10px] text-[#444] shrink-0 pl-2">
                    {item.parentLabel}
                  </span>
                )}
              </button>
            ))
          )
        ) : (
          items.map((item, i) => (
            <div
              key={i}
              className="relative"
              onMouseEnter={() =>
                item.children ? setActiveSubmenu(i) : setActiveSubmenu(null)
              }
              onMouseLeave={() => setActiveSubmenu(null)}
            >
              <button
                onClick={() => {
                  if (!item.children) handleSelect(item);
                }}
                className={`w-full text-left px-3 py-1.5 text-[13px] flex items-center gap-2.5 transition-colors ${
                  item.danger
                    ? "text-red-400 hover:bg-[rgba(255,255,255,0.05)] hover:text-red-300"
                    : "text-[#e0e0e0] hover:bg-[rgba(255,255,255,0.05)]"
                }`}
              >
                {item.icon && (
                  <span className="text-[#777] shrink-0 [&_svg]:w-[15px] [&_svg]:h-[15px]">
                    {item.icon}
                  </span>
                )}
                <span className="flex-1">{item.label}</span>
                {item.children && (
                  <span className="text-[#555] text-[11px] ml-2">
                    {flipX ? "‹" : "›"}
                  </span>
                )}
              </button>

              {item.children && activeSubmenu === i && (
                <SubmenuPanel
                  items={item.children}
                  flipX={flipX}
                  flipY={flipY}
                  onSelect={handleSelect}
                />
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/** Viewport-aware submenu, flips horizontally/vertically when needed. */
function SubmenuPanel({
  items,
  flipX,
  flipY,
  onSelect,
}: {
  items: ContextMenuItem[];
  flipX: boolean;
  flipY: boolean;
  onSelect: (item: ContextMenuItem) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<React.CSSProperties>({
    visibility: "hidden",
    position: "absolute",
  });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Measure the inner visual panel, not the padding wrapper
    const inner = el.firstElementChild as HTMLElement | null;
    if (!inner) return;
    const rect = inner.getBoundingClientRect();
    const parent = el.parentElement?.getBoundingClientRect();
    if (!parent) return;

    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const css: React.CSSProperties = {
      position: "absolute",
      visibility: "visible",
    };

    // Open right by default; flip left if needed.
    // Padding (not margin) keeps the hit area bridging the gap to parent row.
    if (flipX || parent.right + rect.width + 4 > vw - VIEWPORT_PADDING) {
      css.right = "100%";
      css.left = undefined;
      css.paddingRight = "4px";
    } else {
      css.left = "100%";
      css.right = undefined;
      css.paddingLeft = "4px";
    }

    // Align top with parent row, clamp to viewport
    if (flipY || parent.top + rect.height > vh - VIEWPORT_PADDING) {
      css.bottom = 0;
      css.top = undefined;
    } else {
      css.top = 0;
      css.bottom = undefined;
    }

    setStyle(css);
  }, [flipX, flipY]);

  return (
    <div ref={ref} style={style}>
      <div className="min-w-[140px] rounded-lg bg-[#141414] border border-[rgba(255,255,255,0.06)] shadow-[0_4px_24px_rgba(0,0,0,0.5)] py-1.5">
        {items.map((child, ci) => (
          <button
            key={ci}
            onClick={() => onSelect(child)}
            className="w-full text-left px-3 py-1.5 text-[13px] flex items-center gap-2.5 text-[#e0e0e0] hover:bg-[rgba(255,255,255,0.05)] transition-colors"
          >
            {child.icon && (
              <span className="text-[#777] shrink-0 [&_svg]:w-[15px] [&_svg]:h-[15px]">
                {child.icon}
              </span>
            )}
            {child.label}
          </button>
        ))}
      </div>
    </div>
  );
}
