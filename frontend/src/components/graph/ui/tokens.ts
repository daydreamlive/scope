export const NODE_TOKENS = {
  card: "bg-[#242424] border-2 border-transparent rounded-lg shadow-[0_2px_8px_rgba(0,0,0,0.3)] min-w-[240px] relative w-full h-full flex flex-col",
  cardSelected: "[outline:0.5px_solid_white] [outline-offset:3px]",
  header:
    "bg-[#1e1e1e] border-b border-[rgba(255,255,255,0.04)] flex items-center gap-2 px-2.5 py-1 h-[30px] rounded-t-lg",
  body: "py-1 px-2.5",
  bodyWithGap: "py-1 px-2.5 flex flex-col gap-0.5",
  pill: "bg-[#1a1a1a] border border-[rgba(255,255,255,0.06)] rounded-md px-2.5 py-0.5",
  pillInput:
    "bg-[#1a1a1a] border border-[rgba(255,255,255,0.06)] rounded-md px-2.5 py-0.5 text-[#fafafa] text-[10px] appearance-none focus:outline-none focus:ring-1 focus:ring-blue-400/60 w-[110px]",
  pillInputText: "text-center",
  pillInputNumber:
    "text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none",
  labelText: "text-[#737373] text-[10px] font-normal tracking-wide",
  primaryText: "text-[#e8e8e8] text-[10px] font-medium",
  headerText: "text-[#f0f0f0] text-[11px] font-semibold tracking-tight",
  paramRow: "flex items-center justify-between min-h-[18px]",
  sectionTitle:
    "text-[9px] font-semibold text-[#666] uppercase tracking-widest mb-2",
  panelBackground: "bg-[#161616]",
  panelBorder: "border-[rgba(255,255,255,0.04)]",
  toolbar:
    "flex items-stretch h-9 bg-[#161616] border-b border-[rgba(255,255,255,0.06)]",
  toolbarMenuButton:
    "inline-flex items-center gap-1.5 px-4 text-xs font-medium text-[#8c8c8d] hover:bg-[rgba(255,255,255,0.08)] hover:text-[#fafafa] transition-colors cursor-pointer select-none",
  toolbarHeroRun:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 transition-colors cursor-pointer select-none",
  toolbarHeroStop:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold bg-red-600/90 text-white hover:bg-red-600 transition-colors cursor-pointer select-none",
  toolbarHeroBusy:
    "inline-flex items-center gap-1.5 px-5 text-xs font-semibold text-[#8c8c8d] opacity-60 cursor-not-allowed select-none",
  toolbarStatus: "text-xs text-[#8c8c8d] mr-3 self-center",
} as const;
