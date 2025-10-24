// Color palette for prompt boxes
export const PROMPT_COLORS = [
  "#FF6B6B",
  "#4ECDC4",
  "#45B7D1",
  "#96CEB4",
  "#FFEAA7",
  "#DDA0DD",
  "#98D8C8",
  "#F7DC6F",
  "#BB8FCE",
  "#85C1E9",
  "#F8C471",
  "#82E0AA",
  "#F1948A",
  "#85C1E9",
  "#D7BDE2",
];

// Generate random color for prompt boxes
export const generateRandomColor = (excludeColors: string[] = []): string => {
  const availableColors = PROMPT_COLORS.filter(
    color => !excludeColors.includes(color)
  );

  if (availableColors.length === 0) {
    return PROMPT_COLORS[Math.floor(Math.random() * PROMPT_COLORS.length)];
  }

  return availableColors[Math.floor(Math.random() * availableColors.length)];
};
