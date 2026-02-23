import { useState, useEffect } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { AccountTab } from "./settings/AccountTab";
import { ApiKeysTab } from "./settings/ApiKeysTab";
import { GeneralTab } from "./settings/GeneralTab";
import { ReportBugDialog } from "./ReportBugDialog";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import { useLoRAsContext } from "@/contexts/LoRAsContext";
import { LoRAsTab } from "./settings/LoRAsTab";
import { installLoRAFile, deleteLoRAFile, getServerInfo } from "@/lib/api";
import { toast } from "sonner";

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  initialTab?: "general" | "account" | "api-keys" | "loras";
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
}

export function SettingsDialog({
  open,
  onClose,
  initialTab = "general",
  onPipelinesRefresh,
  cloudDisabled,
}: SettingsDialogProps) {
  const { refetch: refetchPipelines } = usePipelinesContext();
  const {
    loraFiles,
    isLoading: isLoadingLoRAs,
    refresh: refreshLoRAs,
  } = useLoRAsContext();
  const [modelsDirectory, setModelsDirectory] = useState(
    "~/.daydream-scope/models"
  );
  const [logsDirectory, setLogsDirectory] = useState("~/.daydream-scope/logs");
  const [reportBugOpen, setReportBugOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(initialTab);
  // LoRA install state (files come from context)
  const [loraInstallUrl, setLoraInstallUrl] = useState("");
  const [isInstallingLoRA, setIsInstallingLoRA] = useState(false);
  const [deletingLoRAs, setDeletingLoRAs] = useState<Set<string>>(new Set());
  const [version, setVersion] = useState<string>("");
  const [gitCommit, setGitCommit] = useState<string>("");

  useEffect(() => {
    if (open) {
      setActiveTab(initialTab);
    }
  }, [open, initialTab]);

  useEffect(() => {
    if (open) {
      getServerInfo()
        .then(info => {
          setVersion(info.version);
          setGitCommit(info.gitCommit);
        })
        .catch(err => console.error("Failed to fetch server info:", err));
    }
  }, [open]);

  // Refresh LoRAs when switching to LoRAs tab
  useEffect(() => {
    if (open && activeTab === "loras") {
      refreshLoRAs();
    }
  }, [open, activeTab, refreshLoRAs]);

  const handleModelsDirectoryChange = (value: string) => {
    console.log("Models directory changed:", value);
    setModelsDirectory(value);
  };

  const handleLogsDirectoryChange = (value: string) => {
    console.log("Logs directory changed:", value);
    setLogsDirectory(value);
  };

  const handleInstallLoRA = async (url: string) => {
    setIsInstallingLoRA(true);
    const filename = url.split("/").pop()?.split("?")[0] || "LoRA file";
    const toastId = toast.loading(`Installing ${filename}...`);
    try {
      const response = await installLoRAFile({ url });
      toast.success(response.message, { id: toastId });
      setLoraInstallUrl("");
      await refreshLoRAs();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Install failed";
      toast.error(message, { id: toastId });
      console.error("Failed to install LoRA:", error);
    } finally {
      setIsInstallingLoRA(false);
    }
  };

  const handleDeleteLoRA = async (name: string) => {
    setDeletingLoRAs(prev => new Set(prev).add(name));
    try {
      const response = await deleteLoRAFile(name);
      if (response.success) {
        toast.success(response.message);
        await refreshLoRAs();
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to delete LoRA"
      );
      console.error("Failed to delete LoRA:", error);
    } finally {
      setDeletingLoRAs(prev => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0">
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
              value="account"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Account
            </TabsTrigger>
            <TabsTrigger
              value="api-keys"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              API Keys
            </TabsTrigger>
            <TabsTrigger
              value="loras"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              LoRAs
            </TabsTrigger>
          </TabsList>
          <div className="w-px bg-border self-stretch" />
          <div className="flex-1 min-w-0 p-4 pt-10 h-[40vh] overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
            <TabsContent value="general" className="mt-0">
              <GeneralTab
                version={version}
                gitCommit={gitCommit}
                modelsDirectory={modelsDirectory}
                logsDirectory={logsDirectory}
                onModelsDirectoryChange={handleModelsDirectoryChange}
                onLogsDirectoryChange={handleLogsDirectoryChange}
                onReportBug={() => setReportBugOpen(true)}
              />
            </TabsContent>
            <TabsContent value="account" className="mt-0">
              <AccountTab
                onPipelinesRefresh={onPipelinesRefresh ?? refetchPipelines}
                cloudDisabled={cloudDisabled}
              />
            </TabsContent>
            <TabsContent value="api-keys" className="mt-0">
              <ApiKeysTab isActive={open && activeTab === "api-keys"} />
            </TabsContent>
            <TabsContent value="loras" className="mt-0">
              <LoRAsTab
                loraFiles={loraFiles}
                installUrl={loraInstallUrl}
                onInstallUrlChange={setLoraInstallUrl}
                onInstall={handleInstallLoRA}
                onDelete={handleDeleteLoRA}
                onRefresh={refreshLoRAs}
                isLoading={isLoadingLoRAs}
                isInstalling={isInstallingLoRA}
                deletingLoRAs={deletingLoRAs}
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
