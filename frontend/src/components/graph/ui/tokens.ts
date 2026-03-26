export const NODE_TOKENS = {
  card: "bg-[#2a2a2a] border border-[rgba(119,119,119,0.55)] rounded-xl min-w-[240px] relative w-full h-full flex flex-col",
  cardSelected: "ring-2 ring-blue-400/50",
  header:
    "bg-[#181717] border-b border-[rgba(119,119,119,0.15)] flex items-center gap-2 px-2 py-1 h-[28px] rounded-t-xl",
  body: "py-1.5 px-2",
  bodyWithGap: "py-1.5 px-2 flex flex-col gap-1.5",
  pill: "bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5",
  pillInput:
    "bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-2 py-0.5 text-[#fafafa] text-[10px] appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/50 w-[110px]",
  pillInputText: "text-center",
  pillInputNumber:
    "text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none",
  labelText: "text-[#8c8c8d] text-[10px] font-normal",
  primaryText: "text-[#fafafa] text-[10px] font-normal",
  headerText: "text-[#fafafa] text-[12px] font-normal",
  paramRow: "flex items-center justify-between h-[20px]",
  sectionTitle: "text-[10px] font-normal text-[#8c8c8d] mb-2",
  panelBackground: "bg-[#181717]",
  panelBorder: "border-[rgba(119,119,119,0.15)]",
  toolbar:
    "flex items-stretch h-9 bg-[#181717] border-b border-[rgba(119,119,119,0.15)]",
  toolbarMenuButton:
    "inline-flex items-center gap-1.5 px-4 text-xs font-medium text-[#8c8c8d] hover:bg-[rgba(255,255,255,0.06)] hover:text-[#fafafa] transition-colors cursor-pointer",
  toolbarHeroRun:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-700 transition-colors cursor-pointer",
  toolbarHeroStop:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold bg-red-600/80 text-white hover:bg-red-700 transition-colors cursor-pointer",
  toolbarHeroBusy:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold text-[#8c8c8d] opacity-60 cursor-not-allowed",
  toolbarStatus: "text-xs text-[#8c8c8d] mr-3 self-center",
} as const;
