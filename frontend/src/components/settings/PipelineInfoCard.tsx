import { Card, CardContent } from "../ui/card";
import { Badge } from "../ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { Hammer, Info } from "lucide-react";
import type { PipelineInfo } from "../../types";

interface PipelineInfoCardProps {
  pipeline: PipelineInfo;
}

export function PipelineInfoCard({ pipeline }: PipelineInfoCardProps) {
  return (
    <Card>
      <CardContent className="p-4 space-y-2">
        <div>
          <h4 className="text-sm font-semibold">
            {pipeline.name}
            {pipeline.pluginName && (
              <span className="font-normal text-muted-foreground">
                {" "}
                ({pipeline.pluginName})
              </span>
            )}
          </h4>
        </div>

        <div>
          {(pipeline.about || pipeline.docsUrl || pipeline.modified) && (
            <div className="flex items-stretch gap-1 h-6">
              {pipeline.about && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge
                        variant="outline"
                        className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                      >
                        <Info className="h-3.5 w-3.5" />
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="text-xs">{pipeline.about}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              {pipeline.modified && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge
                        variant="outline"
                        className="cursor-help hover:bg-accent h-full flex items-center justify-center"
                      >
                        <Hammer className="h-3.5 w-3.5" />
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        This pipeline contains modifications based on the
                        original project.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              {pipeline.docsUrl && (
                <a
                  href={pipeline.docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block h-full"
                >
                  <Badge
                    variant="outline"
                    className="hover:bg-accent cursor-pointer h-full flex items-center"
                  >
                    Docs
                  </Badge>
                </a>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
