import { BookOpenText } from "lucide-react";

interface HeaderProps {
  className?: string;
}

export function Header({ className = "" }: HeaderProps) {
  return (
    <header className={`w-full bg-background px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-medium text-foreground">Daydream Scope</h1>
        <div className="flex items-center gap-3">
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
    </header>
  );
}
