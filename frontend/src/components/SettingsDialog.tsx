import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { GeneralTab } from "./settings/GeneralTab";
import { PluginsTab } from "./settings/PluginsTab";
import { ReportBugDialog } from "./ReportBugDialog";
import { MOCK_PLUGINS } from "@/types/settings";

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const [modelsDirectory, setModelsDirectory] = useState(
    "~/.daydream-scope/models"
  );
  const [logsDirectory, setLogsDirectory] = useState("~/.daydream-scope/logs");
  const [reportBugOpen, setReportBugOpen] = useState(false);
  const [pluginInstallPath, setPluginInstallPath] = useState("");

  const handleModelsDirectoryChange = (value: string) => {
    console.log("Models directory changed:", value);
    setModelsDirectory(value);
  };

  const handleLogsDirectoryChange = (value: string) => {
    console.log("Logs directory changed:", value);
    setLogsDirectory(value);
  };

  const handleBrowsePlugin = () => {
    console.log("Browse for plugin clicked");
    // TODO: Open file dialog via Electron IPC
  };

  const handleInstallPlugin = (pluginUrl: string) => {
    console.log("Install plugin:", pluginUrl);
    setPluginInstallPath("");
  };

  const handleCheckUpdates = () => {
    console.log("Check for updates clicked");
  };

  const handleDeletePlugin = (pluginId: string) => {
    console.log("Delete plugin:", pluginId);
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0">
        <DialogHeader className="sr-only">
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>
        <Tabs
          defaultValue="general"
          orientation="vertical"
          className="flex items-stretch"
        >
          <TabsList className="flex flex-col items-start justify-start bg-transparent gap-1 w-32 p-4">
            <TabsTrigger
              value="general"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              General
            </TabsTrigger>
            <TabsTrigger
              value="plugins"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Plugins
            </TabsTrigger>
          </TabsList>
          <div className="w-px bg-border self-stretch" />
          <div className="flex-1 min-w-0 p-4 pt-10 h-[40vh] overflow-y-auto">
            <TabsContent value="general" className="mt-0">
              <GeneralTab
                version="0.1.0"
                modelsDirectory={modelsDirectory}
                logsDirectory={logsDirectory}
                onModelsDirectoryChange={handleModelsDirectoryChange}
                onLogsDirectoryChange={handleLogsDirectoryChange}
                onReportBug={() => setReportBugOpen(true)}
              />
            </TabsContent>
            <TabsContent value="plugins" className="mt-0">
              <PluginsTab
                plugins={MOCK_PLUGINS}
                installPath={pluginInstallPath}
                onInstallPathChange={setPluginInstallPath}
                onBrowse={handleBrowsePlugin}
                onInstall={handleInstallPlugin}
                onCheckUpdates={handleCheckUpdates}
                onDelete={handleDeletePlugin}
              />
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>

      <ReportBugDialog
        open={reportBugOpen}
        onClose={() => setReportBugOpen(false)}
      />
    </Dialog>
  );
}
