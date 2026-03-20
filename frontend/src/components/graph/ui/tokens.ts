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
    "flex items-center gap-2 px-4 py-2 bg-[#181717] border-b border-[rgba(119,119,119,0.15)]",
  toolbarButton:
    "px-3 py-1.5 text-xs font-medium rounded-lg bg-[#2a2a2a] border border-[rgba(119,119,119,0.35)] text-[#fafafa] hover:bg-[#2a2a2a]/80 transition-colors",
  toolbarStatus: "text-xs text-[#8c8c8d] ml-2",
} as const;
