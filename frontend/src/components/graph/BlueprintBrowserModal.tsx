import { useState, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { BLUEPRINTS } from "../../data/blueprints";
import type { Blueprint } from "../../data/blueprints/types";

interface BlueprintBrowserModalProps {
  open: boolean;
  onClose: () => void;
  onInsert: (blueprint: Blueprint) => void;
}

const ALL_CATEGORIES = [
  "All",
  ...Array.from(new Set(BLUEPRINTS.map(b => b.category))).sort(),
];

function BlueprintCard({
  blueprint,
  onInsert,
}: {
  blueprint: Blueprint;
  onInsert: () => void;
}) {
  return (
    <div className="flex flex-col gap-2 p-3 rounded-xl bg-[#242424] border border-[rgba(119,119,119,0.12)] hover:border-[rgba(119,119,119,0.3)] transition-colors">
      <div className="flex items-center gap-2">
        <div
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: blueprint.color }}
        />
        <span className="text-[12px] font-medium text-[#e0e0e0] truncate flex-1">
          {blueprint.name}
        </span>
        <span className="text-[10px] text-[#666] shrink-0 bg-[#1a1a1a] px-1.5 py-0.5 rounded-full border border-[rgba(119,119,119,0.12)]">
          {blueprint.category}
        </span>
      </div>
      <p className="text-[11px] text-[#888] leading-relaxed line-clamp-2">
        {blueprint.description}
      </p>
      <div className="flex items-center justify-between mt-1">
        <span className="text-[10px] text-[#555]">
          {blueprint.nodes.length} node{blueprint.nodes.length !== 1 ? "s" : ""}
          {blueprint.edges.length > 0 &&
            `, ${blueprint.edges.length} edge${blueprint.edges.length !== 1 ? "s" : ""}`}
        </span>
        <button
          onClick={onInsert}
          className="px-3 py-1 rounded-full bg-[#fafafa] text-[#111] text-[11px] font-medium hover:bg-[#e0e0e0] transition-colors"
        >
          Insert
        </button>
      </div>
    </div>
  );
}

export function BlueprintBrowserModal({
  open,
  onClose,
  onInsert,
}: BlueprintBrowserModalProps) {
  const [searchText, setSearchText] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const filteredBlueprints = useMemo(() => {
    const lower = searchText.toLowerCase();
    return BLUEPRINTS.filter(b => {
      const matchesSearch =
        !lower ||
        b.name.toLowerCase().includes(lower) ||
        b.description.toLowerCase().includes(lower) ||
        b.category.toLowerCase().includes(lower);
      const matchesCategory =
        activeCategory === "All" || b.category === activeCategory;
      return matchesSearch && matchesCategory;
    });
  }, [searchText, activeCategory]);

  const handleInsert = (blueprint: Blueprint) => {
    onInsert(blueprint);
    handleClose();
  };

  const handleClose = () => {
    onClose();
    setSearchText("");
    setActiveCategory("All");
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="!max-w-xl w-full p-0 overflow-hidden bg-[#1a1a1a] border border-[rgba(119,119,119,0.2)] rounded-2xl">
        <DialogHeader className="sr-only">
          <DialogTitle>Blueprint Library</DialogTitle>
          <DialogDescription>
            Browse and insert pre-made node groups into your workflow
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col h-[480px]">
          {/* Header */}
          <div className="px-4 pt-4 pb-3 border-b border-[rgba(119,119,119,0.12)]">
            <div className="flex items-center gap-2 px-3 py-2 bg-[#111] rounded-lg border border-[rgba(119,119,119,0.2)]">
              <svg
                className="w-3.5 h-3.5 text-[#666] shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z"
                />
              </svg>
              <input
                type="text"
                value={searchText}
                onChange={e => setSearchText(e.target.value)}
                placeholder="Search blueprints..."
                className="flex-1 bg-transparent text-xs text-[#fafafa] placeholder:text-[#555] focus:outline-none"
                autoFocus
              />
            </div>
          </div>

          {/* Category tabs */}
          <div className="flex items-center gap-1.5 px-4 py-2 border-b border-[rgba(119,119,119,0.12)] flex-wrap">
            {ALL_CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  activeCategory === cat
                    ? "bg-[#fafafa] text-[#111]"
                    : "text-[#888] hover:text-[#ccc]"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Blueprint list */}
          <div className="flex-1 overflow-y-auto px-4 py-3 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded-full">
            {filteredBlueprints.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#555] text-xs">
                No blueprints found
                {searchText ? ` for "${searchText}"` : ""}
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {filteredBlueprints.map(blueprint => (
                  <BlueprintCard
                    key={blueprint.name}
                    blueprint={blueprint}
                    onInsert={() => handleInsert(blueprint)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-[rgba(119,119,119,0.12)]">
            <button
              onClick={handleClose}
              className="px-5 py-2 rounded-full bg-[#2a2a2a] border border-[rgba(119,119,119,0.2)] text-xs font-medium text-[#fafafa] hover:bg-[#333] transition-colors"
            >
              Cancel
            </button>
            <span className="text-[10px] text-[#555]">
              {filteredBlueprints.length} blueprint
              {filteredBlueprints.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
