import { useState, useEffect, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import { Button } from "../ui/button";
import type { TourStepDef } from "./tourSteps";

// ---------------------------------------------------------------------------
// Positioning helpers
// ---------------------------------------------------------------------------

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const POPOVER_WIDTH = 320;
const POPOVER_GAP = 12;
const ARROW_SIZE = 8;

function getPopoverStyle(
  anchorRect: Rect | null,
  position: TourStepDef["position"]
): React.CSSProperties {
  // Centered (no anchor)
  if (!anchorRect || position === "center") {
    return {
      position: "fixed",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      width: POPOVER_WIDTH,
    };
  }

  const style: React.CSSProperties = {
    position: "fixed",
    width: POPOVER_WIDTH,
  };

  switch (position) {
    case "bottom":
      style.top = anchorRect.top + anchorRect.height + POPOVER_GAP;
      style.left = anchorRect.left + anchorRect.width / 2 - POPOVER_WIDTH / 2;
      break;
    case "top":
      style.bottom = window.innerHeight - anchorRect.top + POPOVER_GAP;
      style.left = anchorRect.left + anchorRect.width / 2 - POPOVER_WIDTH / 2;
      break;
    case "left":
      style.top = anchorRect.top + anchorRect.height / 2 - 40; // rough vertical centering
      style.right = window.innerWidth - anchorRect.left + POPOVER_GAP;
      break;
    case "right":
      style.top = anchorRect.top + anchorRect.height / 2 - 40;
      style.left = anchorRect.left + anchorRect.width + POPOVER_GAP;
      break;
  }

  // Clamp to viewport
  if (typeof style.left === "number") {
    style.left = Math.max(
      12,
      Math.min(style.left, window.innerWidth - POPOVER_WIDTH - 12)
    );
  }

  return style;
}

function getArrowStyle(
  anchorRect: Rect | null,
  position: TourStepDef["position"],
  popoverStyle: React.CSSProperties
): React.CSSProperties | null {
  if (!anchorRect || position === "center") return null;

  const base: React.CSSProperties = {
    position: "absolute",
    width: 0,
    height: 0,
  };

  // For top/bottom positions, compute horizontal offset so the arrow points
  // at the anchor center even when the popover was clamped to the viewport.
  const anchorCenterX = anchorRect.left + anchorRect.width / 2;
  const popoverLeft = typeof popoverStyle.left === "number" ? popoverStyle.left : 0;
  const arrowLeft = Math.max(
    ARROW_SIZE + 8,
    Math.min(anchorCenterX - popoverLeft, POPOVER_WIDTH - ARROW_SIZE - 8)
  );

  switch (position) {
    case "bottom":
      return {
        ...base,
        top: -ARROW_SIZE,
        left: arrowLeft,
        transform: "translateX(-50%)",
        borderLeft: `${ARROW_SIZE}px solid transparent`,
        borderRight: `${ARROW_SIZE}px solid transparent`,
        borderBottom: `${ARROW_SIZE}px solid rgba(119,119,119,0.2)`,
      };
    case "top":
      return {
        ...base,
        bottom: -ARROW_SIZE,
        left: arrowLeft,
        transform: "translateX(-50%)",
        borderLeft: `${ARROW_SIZE}px solid transparent`,
        borderRight: `${ARROW_SIZE}px solid transparent`,
        borderTop: `${ARROW_SIZE}px solid rgba(119,119,119,0.2)`,
      };
    case "left":
      return {
        ...base,
        right: -ARROW_SIZE,
        top: 20,
        borderTop: `${ARROW_SIZE}px solid transparent`,
        borderBottom: `${ARROW_SIZE}px solid transparent`,
        borderLeft: `${ARROW_SIZE}px solid rgba(119,119,119,0.2)`,
      };
    case "right":
      return {
        ...base,
        left: -ARROW_SIZE,
        top: 20,
        borderTop: `${ARROW_SIZE}px solid transparent`,
        borderBottom: `${ARROW_SIZE}px solid transparent`,
        borderRight: `${ARROW_SIZE}px solid rgba(119,119,119,0.2)`,
      };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Spotlight overlay
// ---------------------------------------------------------------------------

function SpotlightOverlay({ anchorRect }: { anchorRect: Rect | null }) {
  if (!anchorRect) {
    // Semi-transparent full screen for centered popovers
    return (
      <div
        className="fixed inset-0 bg-black/20 pointer-events-none"
        style={{ zIndex: 89 }}
      />
    );
  }

  // Box-shadow trick: a huge box-shadow creates the dim effect with a cutout
  const pad = 6;
  return (
    <div
      className="fixed pointer-events-none"
      style={{
        zIndex: 89,
        top: anchorRect.top - pad,
        left: anchorRect.left - pad,
        width: anchorRect.width + pad * 2,
        height: anchorRect.height + pad * 2,
        borderRadius: 8,
        boxShadow: "0 0 0 9999px rgba(0,0,0,0.20)",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// TourPopover
// ---------------------------------------------------------------------------

interface TourPopoverProps {
  step: TourStepDef;
  stepIndex: number;
  totalSteps: number;
  onNext: () => void;
  onSkip: () => void;
}

export function TourPopover({
  step,
  stepIndex,
  totalSteps,
  onNext,
  onSkip,
}: TourPopoverProps) {
  const [anchorRect, setAnchorRect] = useState<Rect | null>(null);
  const rafRef = useRef<number | null>(null);

  const updatePosition = useCallback(() => {
    if (!step.anchor) {
      setAnchorRect(null);
      return;
    }

    let el = document.querySelector(`[data-tour="${step.anchor}"]`);
    if (!el && step.fallbackAnchor) {
      el = document.querySelector(`[data-tour="${step.fallbackAnchor}"]`);
    }

    if (el) {
      const rect = el.getBoundingClientRect();
      setAnchorRect({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
    } else {
      setAnchorRect(null);
    }
  }, [step.anchor, step.fallbackAnchor]);

  // Recalculate on mount, resize, scroll
  useEffect(() => {
    updatePosition();

    const handleUpdate = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(updatePosition);
    };

    window.addEventListener("resize", handleUpdate);
    window.addEventListener("scroll", handleUpdate, true);

    // Also poll briefly in case layout hasn't settled yet
    const pollTimer = setInterval(updatePosition, 200);
    const stopPoll = setTimeout(() => clearInterval(pollTimer), 2000);

    return () => {
      window.removeEventListener("resize", handleUpdate);
      window.removeEventListener("scroll", handleUpdate, true);
      clearInterval(pollTimer);
      clearTimeout(stopPoll);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [updatePosition]);

  const popoverStyle = getPopoverStyle(anchorRect, step.position);
  const arrowStyle = getArrowStyle(anchorRect, step.position, popoverStyle);

  return createPortal(
    <>
      {/* Spotlight dim */}
      <SpotlightOverlay anchorRect={anchorRect} />

      {/* Popover card */}
      <div
        role="dialog"
        aria-label={`Tour step ${stepIndex + 1} of ${totalSteps}: ${step.title}`}
        className="animate-in fade-in-0 zoom-in-95 duration-200"
        style={{ ...popoverStyle, zIndex: 90 }}
      >
        <div className="bg-[#1a1a1a] border border-[rgba(119,119,119,0.2)] rounded-xl p-4 shadow-xl relative">
          {/* Arrow */}
          {arrowStyle && <div style={arrowStyle} />}

          {/* Content */}
          <div className="space-y-3">
            <p className="text-sm font-medium text-[#fafafa]">{step.title}</p>
            <p className="text-xs text-[#aaa] leading-relaxed">
              {step.description}
              {step.linkUrl && (
                <>
                  {" "}
                  <a
                    href={step.linkUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[#60a5fa] hover:underline"
                  >
                    {step.linkText ?? step.linkUrl}
                  </a>
                </>
              )}
            </p>

            {/* Footer */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-[10px] text-[#666]">
                {stepIndex + 1} of {totalSteps}
              </span>
              <div className="flex items-center gap-3">
                {step.showSkip && (
                  <button
                    onClick={onSkip}
                    className="text-[11px] text-[#888] hover:text-[#ccc] transition-colors"
                  >
                    Skip tour
                  </button>
                )}
                <Button onClick={onNext} size="sm" className="h-7 px-4 text-xs">
                  {step.showDone ? "Done" : "Next"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>,
    document.body
  );
}
