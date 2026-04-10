import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import type {
  ScopeWorkflow,
  ResolutionItem,
  WorkflowLoRAProvenance,
} from "../lib/workflowApi";

export const statusIcon = (status: ResolutionItem["status"]) => {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />;
    case "missing":
      return <XCircle className="h-4 w-4 text-red-500 shrink-0" />;
    case "version_mismatch":
      return <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />;
  }
};

export const kindLabel = (kind: ResolutionItem["kind"]) => {
  switch (kind) {
    case "pipeline":
      return "Model";
    case "plugin":
      return "Node";
    case "lora":
      return "LoRA";
  }
};

export function provenanceLabel(prov: WorkflowLoRAProvenance): string {
  if (prov.source === "huggingface" && prov.repo_id) {
    return `HuggingFace: ${prov.repo_id}`;
  }
  if (prov.source === "civitai") {
    return `CivitAI model ${prov.model_id ?? prov.version_id ?? ""}`;
  }
  if (prov.source === "url" && prov.url) {
    return prov.url;
  }
  return prov.source;
}

export function findLoRAProvenance(
  workflow: ScopeWorkflow,
  filename: string
): WorkflowLoRAProvenance | null {
  const lora = workflow.pipelines
    .flatMap(p => p.loras)
    .find(l => l.filename === filename);
  if (lora?.provenance && lora.provenance.source !== "local") {
    return lora.provenance;
  }
  return null;
}

export function LoRAProvenanceLabel({
  workflow,
  filename,
}: {
  workflow: ScopeWorkflow;
  filename: string;
}) {
  const prov = findLoRAProvenance(workflow, filename);
  if (!prov) return null;
  return (
    <p className="text-[10px] text-muted-foreground mt-0.5">
      {provenanceLabel(prov)}
    </p>
  );
}
