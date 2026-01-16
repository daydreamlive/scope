/**
 * Shared types for dynamic control components.
 */

import type { ResolvedSchemaProperty } from "../../lib/schemaInference";

/**
 * Base props for all dynamic control components.
 * Uses a generic type parameter T for type-safe value handling.
 */
export interface DynamicControlProps<T = unknown> {
  /** Parameter name (used as key for updates) */
  paramName: string;
  /** Resolved JSON schema property with constraints */
  schema: ResolvedSchemaProperty;
  /** Current value */
  value: T;
  /** Callback when value changes */
  onChange: (value: T) => void;
  /** Display label */
  label: string;
  /** Optional tooltip description */
  tooltip?: string;
  /** Whether the control is disabled */
  disabled?: boolean;
}

/**
 * Props for slider control (bounded numeric values).
 */
export type SliderControlProps = DynamicControlProps<number>;

/**
 * Props for number input control (unbounded or partially bounded numeric values).
 */
export type NumberControlProps = DynamicControlProps<number>;

/**
 * Props for toggle control (boolean values).
 */
export type ToggleControlProps = DynamicControlProps<boolean>;

/**
 * Props for select control (enum values).
 */
export interface SelectControlProps extends DynamicControlProps<string> {
  options: string[];
}

/**
 * Props for text input control (string values).
 */
export type TextControlProps = DynamicControlProps<string>;
