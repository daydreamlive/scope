import { BookOpenText, Bug, Github } from "lucide-react";
import { Input } from "../ui/input";

interface GeneralTabProps {
  version: string;
  modelsDirectory: string;
  logsDirectory: string;
  onModelsDirectoryChange: (value: string) => void;
  onLogsDirectoryChange: (value: string) => void;
  onReportBug: () => void;
}

export function GeneralTab({
  version,
  modelsDirectory,
  logsDirectory,
  onModelsDirectoryChange,
  onLogsDirectoryChange,
  onReportBug,
}: GeneralTabProps) {
  const handleDocsClick = () => {
    console.log("Docs clicked");
    window.open(
      "https://docs.daydream.live/knowledge-hub/tutorials/scope",
      "_blank"
    );
  };

  const handleDiscordClick = () => {
    console.log("Discord clicked");
    window.open("https://discord.gg/mnfGR4Fjhp", "_blank");
  };

  const handleGithubClick = () => {
    console.log("GitHub clicked");
    window.open("https://github.com/daydreamlive/scope", "_blank");
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Version Info */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Version
          </span>
          <div className="flex-1 flex items-center justify-end">
            <span className="text-sm text-muted-foreground">{version}</span>
          </div>
        </div>

        {/* Help */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">Help</span>
          <div className="flex-1 flex items-center justify-end gap-1">
            <button
              onClick={onReportBug}
              className="flex items-center gap-1.5 p-2 rounded-md hover:bg-accent transition-colors text-muted-foreground"
              title="Report Bug"
            >
              <Bug className="h-5 w-5" />
              <span className="text-xs">Report Bug</span>
            </button>
            <button
              onClick={handleDocsClick}
              className="p-2 rounded-md hover:bg-accent transition-colors"
              title="Documentation"
            >
              <BookOpenText className="h-5 w-5 text-muted-foreground" />
            </button>
            <button
              onClick={handleDiscordClick}
              className="p-2 rounded-md hover:bg-accent transition-colors"
              title="Discord"
            >
              <img
                src="/assets/discord-symbol-white.svg"
                alt="Discord"
                className="h-5 w-5 opacity-60"
              />
            </button>
            <button
              onClick={handleGithubClick}
              className="p-2 rounded-md hover:bg-accent transition-colors"
              title="GitHub"
            >
              <Github className="h-5 w-5 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* Server URL */}
        <div className="flex items-center gap-4">
          <label
            htmlFor="server-url"
            className="text-sm font-medium text-foreground whitespace-nowrap w-32"
          >
            Server URL
          </label>
          <Input
            id="server-url"
            value={window.location.origin}
            readOnly
            className="flex-1"
            disabled
          />
        </div>

        {/* Models Directory */}
        <div className="flex items-center gap-4">
          <label
            htmlFor="models-directory"
            className="text-sm font-medium text-foreground whitespace-nowrap w-32"
          >
            Models Directory
          </label>
          <Input
            id="models-directory"
            value={modelsDirectory}
            onChange={e => onModelsDirectoryChange(e.target.value)}
            placeholder="~/.daydream-scope/models"
            className="flex-1"
            disabled
          />
        </div>

        {/* Logs Directory */}
        <div className="flex items-center gap-4">
          <label
            htmlFor="logs-directory"
            className="text-sm font-medium text-foreground whitespace-nowrap w-32"
          >
            Logs Directory
          </label>
          <Input
            id="logs-directory"
            value={logsDirectory}
            onChange={e => onLogsDirectoryChange(e.target.value)}
            placeholder="~/.daydream-scope/logs"
            className="flex-1"
            disabled
          />
        </div>
      </div>
    </div>
  );
}
