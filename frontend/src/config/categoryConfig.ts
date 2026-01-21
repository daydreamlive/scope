/**
 * Category configuration for dynamic settings panel.
 *
 * Defines section titles, icons, order, and section-level visibility conditions.
 * This is now provided by the backend via schema metadata (category_config).
 * Falls back to defaults if not provided by backend.
 */

import type { ConditionExpression, InputMode } from "../types";
import type { CategoryConfig as APICategoryConfig } from "../lib/api";

/**
 * Configuration for a category/section in the settings panel.
 */
export interface CategoryConfig {
  /** Display title for the section */
  title: string;
  /** Optional icon name (for future icon support) */
  icon?: string;
  /** Display order (lower numbers appear first) */
  order: number;
  /** Whether this section is collapsible */
  collapsible?: boolean;
  /** Condition expression for section-level visibility */
  showIf?: ConditionExpression;
  /** Modes in which this section should be hidden */
  hideInModes?: InputMode[];
}

/**
 * Default category configuration (fallback if backend doesn't provide).
 */
const DEFAULT_CATEGORY_CONFIG: Record<string, CategoryConfig> = {
  resolution: {
    title: "Resolution",
    icon: "ruler",
    order: 1,
  },
  generation: {
    title: "Generation",
    order: 2,
  },
  noise: {
    title: "Noise Control",
    order: 3,
  },
  vace: {
    title: "VACE",
    order: 4,
    collapsible: true,
  },
  lora: {
    title: "LoRA",
    order: 5,
  },
  cache: {
    title: "Cache",
    order: 6,
  },
  advanced: {
    title: "Advanced",
    order: 7,
    collapsible: true,
  },
  output: {
    title: "Output",
    order: 8,
  },
};

/**
 * Gets category configuration for a category name.
 * Uses backend-provided config if available, otherwise falls back to defaults.
 */
export function getCategoryConfig(
  categoryName: string,
  backendConfig?: Record<string, APICategoryConfig>
): CategoryConfig {
  // Try backend config first
  if (backendConfig && categoryName in backendConfig) {
    const backend = backendConfig[categoryName];
    return {
      title: backend.title,
      icon: backend.icon,
      order: backend.order,
      collapsible: backend.collapsible,
      showIf: backend.showIf as ConditionExpression | undefined,
      hideInModes: backend.hideInModes as InputMode[] | undefined,
    };
  }

  // Fall back to default config
  if (categoryName in DEFAULT_CATEGORY_CONFIG) {
    return DEFAULT_CATEGORY_CONFIG[categoryName];
  }

  // Generate default for unknown categories
  return {
    title: categoryName
      .split("_")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" "),
    order: 999, // Default to end
  };
}

/**
 * Checks if a category section should be visible.
 * Note: evaluateCondition should be imported from conditionEvaluator when used.
 */
export function shouldShowCategory(
  categoryName: string,
  fieldValues: Record<string, unknown>,
  inputMode?: InputMode,
  evaluateConditionFn?: (
    condition: ConditionExpression,
    fieldValues: Record<string, unknown>,
    inputMode?: InputMode
  ) => boolean
): boolean {
  const config = getCategoryConfig(categoryName);

  // Check if hidden in current mode
  if (
    inputMode &&
    config.hideInModes &&
    config.hideInModes.includes(inputMode)
  ) {
    return false;
  }

  // Evaluate showIf condition
  if (config.showIf && evaluateConditionFn) {
    return evaluateConditionFn(config.showIf, fieldValues, inputMode);
  }

  return true;
}
