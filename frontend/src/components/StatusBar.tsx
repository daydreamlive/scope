import { Terminal } from "lucide-react";

interface StatusBarProps {
  className?: string;
  onLogToggle?: () => void;
  isLogOpen?: boolean;
  logUnreadCount?: number;
}

export function StatusBar({
  className = "",
  onLogToggle,
  isLogOpen,
  logUnreadCount = 0,
}: StatusBarProps) {
  return (
    <div
      className={`border-t bg-muted/30 px-6 py-2 flex items-center flex-shrink-0 ${className}`}
    >
      {/* Left: Log toggle */}
      <div className="flex items-center gap-2">
        {onLogToggle && (
          <button
            onClick={onLogToggle}
            className={`flex items-center gap-1 text-xs transition-colors ${
              isLogOpen
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            title="Toggle log panel"
          >
            <Terminal className="h-3.5 w-3.5" />
            <span>Logs</span>
            {logUnreadCount > 0 && !isLogOpen && (
              <span className="bg-blue-500 text-white text-[10px] px-1 rounded-full min-w-[16px] text-center leading-4">
                {logUnreadCount > 99 ? "99+" : logUnreadCount}
              </span>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
