import { useState, useEffect, useCallback } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { GeneralTab } from "./settings/GeneralTab";
import { PluginsTab } from "./settings/PluginsTab";
import { ReportBugDialog } from "./ReportBugDialog";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import type { InstalledPlugin } from "@/types/settings";
import { listPlugins, installPlugin, uninstallPlugin } from "@/lib/api";
import { toast } from "sonner";

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const { refetch: refetchPipelines } = usePipelinesContext();
  const [modelsDirectory, setModelsDirectory] = useState(
    "~/.daydream-scope/models"
  );
  const [logsDirectory, setLogsDirectory] = useState("~/.daydream-scope/logs");
  const [reportBugOpen, setReportBugOpen] = useState(false);
  const [pluginInstallPath, setPluginInstallPath] = useState("");
  const [plugins, setPlugins] = useState<InstalledPlugin[]>([]);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [activeTab, setActiveTab] = useState("general");

  const fetchPlugins = useCallback(async () => {
    setIsLoadingPlugins(true);
    try {
      const response = await listPlugins();
      setPlugins(
        response.plugins.map(p => ({
          name: p.name,
          version: p.version,
          author: p.author,
          description: p.description,
          source: p.source,
          editable: p.editable,
          latest_version: p.latest_version,
          update_available: p.update_available,
          package_spec: p.package_spec,
        }))
      );
    } catch (error) {
      console.error("Failed to fetch plugins:", error);
      toast.error("Failed to load plugins");
    } finally {
      setIsLoadingPlugins(false);
    }
  }, []);

  // Fetch plugins when dialog opens or when switching to plugins tab
  useEffect(() => {
    if (open && activeTab === "plugins") {
      fetchPlugins();
    }
  }, [open, activeTab, fetchPlugins]);

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

  const handleInstallPlugin = async (packageSpec: string) => {
    setIsInstalling(true);
    try {
      const response = await installPlugin({ package: packageSpec });
      if (response.success) {
        const pluginName = response.plugin?.name || packageSpec;
        toast.success(`Successfully installed ${pluginName}`);
        setPluginInstallPath("");
        await fetchPlugins();
        await refetchPipelines();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to install plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to install plugin"
      );
    } finally {
      setIsInstalling(false);
    }
  };

  const handleUpdatePlugin = async (
    pluginName: string,
    packageSpec: string
  ) => {
    setIsInstalling(true);
    try {
      const response = await installPlugin({
        package: packageSpec,
        upgrade: true,
      });
      if (response.success) {
        toast.success(`Successfully updated ${pluginName}`);
        await fetchPlugins();
        await refetchPipelines();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to update plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to update plugin"
      );
    } finally {
      setIsInstalling(false);
    }
  };

  const handleDeletePlugin = async (pluginName: string) => {
    console.log("Uninstalling plugin:", pluginName);
    try {
      const response = await uninstallPlugin(pluginName);
      console.log("Uninstall response:", response);
      if (response.success) {
        toast.success(response.message);
        await fetchPlugins();
        await refetchPipelines();
      } else {
        console.error("Uninstall failed:", response.message);
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to uninstall plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to uninstall plugin"
      );
    }
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0">
        <DialogHeader className="sr-only">
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
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
                plugins={plugins}
                installPath={pluginInstallPath}
                onInstallPathChange={setPluginInstallPath}
                onBrowse={handleBrowsePlugin}
                onInstall={handleInstallPlugin}
                onUpdate={handleUpdatePlugin}
                onDelete={handleDeletePlugin}
                isLoading={isLoadingPlugins}
                isInstalling={isInstalling}
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
