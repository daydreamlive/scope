import { ChevronRight } from "lucide-react";

interface BreadcrumbNavProps {
  /** Path segments, e.g. ["Root", "MySubgraph", "InnerGroup"]. */
  path: string[];
  /** Called when a segment is clicked. Index 0 = root. */
  onNavigate: (depth: number) => void;
}

export function BreadcrumbNav({ path, onNavigate }: BreadcrumbNavProps) {
  if (path.length <= 1) return null; // Only visible when inside a subgraph

  return (
    <div className="flex items-center gap-1 px-4 py-1.5 bg-[#1a1a1a] border-b border-[rgba(119,119,119,0.15)] text-[11px]">
      {path.map((segment, i) => {
        const isLast = i === path.length - 1;
        return (
          <span key={i} className="flex items-center gap-1">
            {i > 0 && <ChevronRight className="h-3 w-3 text-[#555] shrink-0" />}
            {isLast ? (
              <span className="text-cyan-400 font-medium">{segment}</span>
            ) : (
              <button
                type="button"
                onClick={() => onNavigate(i)}
                className="text-[#999] hover:text-[#fafafa] transition-colors cursor-pointer"
              >
                {segment}
              </button>
            )}
          </span>
        );
      })}
    </div>
  );
}
