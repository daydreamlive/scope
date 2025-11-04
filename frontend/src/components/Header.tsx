import { useState } from "react";
import { BookOpenText, Bug } from "lucide-react";
import { Button } from "./ui/button";
import { ReportBugDialog } from "./ReportBugDialog";

interface HeaderProps {
  className?: string;
}

export function Header({ className = "" }: HeaderProps) {
  const [reportBugOpen, setReportBugOpen] = useState(false);

  return (
    <header className={`w-full bg-background px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-medium text-foreground">Daydream Scope</h1>
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setReportBugOpen(true)}
            className="hover:opacity-80 transition-opacity gap-1.5 text-muted-foreground opacity-60"
          >
            <Bug className="h-4 w-4" />
            <span className="text-xs">Report Bug</span>
          </Button>
          <a
            href="https://github.com/daydreamlive/scope"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <img
              src="/assets/github-mark-white.svg"
              alt="GitHub"
              className="h-5 w-5 opacity-60"
            />
          </a>
          <a
            href="https://discord.gg/mnfGR4Fjhp"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <img
              src="/assets/discord-symbol-white.svg"
              alt="Discord"
              className="h-5 w-5 opacity-60"
            />
          </a>
          <a
            href="https://docs.daydream.live/knowledge-hub/tutorials/scope"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <BookOpenText className="h-5 w-5 text-muted-foreground opacity-60" />
          </a>
        </div>
      </div>

      <ReportBugDialog
        open={reportBugOpen}
        onClose={() => setReportBugOpen(false)}
      />
    </header>
  );
}
