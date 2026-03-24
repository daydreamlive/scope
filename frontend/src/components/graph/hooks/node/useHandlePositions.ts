import { useCallback, useLayoutEffect, useRef, useState } from "react";

// Measures DOM row positions for handle placement. Pass deps to re-measure on changes.
export function useHandlePositions(deps: unknown[] = []) {
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [rowPositions, setRowPositions] = useState<Record<string, number>>({});

  const setRowRef = useCallback(
    (key: string) => (el: HTMLDivElement | null) => {
      if (el) rowRefs.current.set(key, el);
      else rowRefs.current.delete(key);
    },
    []
  );

  useLayoutEffect(() => {
    const positions: Record<string, number> = {};
    for (const [key, el] of rowRefs.current.entries()) {
      positions[key] = el.offsetTop + el.offsetHeight / 2;
    }
    setRowPositions(prev => {
      const keys = Object.keys(positions);
      if (
        keys.length === Object.keys(prev).length &&
        keys.every(k => Math.abs((prev[k] ?? 0) - positions[k]) < 1)
      ) {
        return prev; // no change
      }
      return positions;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { setRowRef, rowPositions };
}
