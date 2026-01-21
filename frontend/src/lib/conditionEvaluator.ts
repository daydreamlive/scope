/**
 * Condition evaluator for dynamic field visibility.
 *
 * Evaluates condition expressions defined in ui:showIf metadata.
 * Supports:
 * - Simple field comparisons (eq, ne, gt, gte, lt, lte, in, nin)
 * - Compound logic (allOf, anyOf)
 * - Negation (not)
 */

import type { ConditionExpression, InputMode } from "../types";

/**
 * Evaluates a condition expression against current field values.
 *
 * @param condition - The condition expression to evaluate
 * @param fieldValues - Map of field names to their current values
 * @param inputMode - Current input mode (for mode-based conditions)
 * @returns true if condition is satisfied, false otherwise
 */
export function evaluateCondition(
  condition: ConditionExpression,
  fieldValues: Record<string, unknown>,
  inputMode?: InputMode
): boolean {
  // Handle compound conditions
  if ("allOf" in condition) {
    return condition.allOf.every(expr =>
      evaluateCondition(expr, fieldValues, inputMode)
    );
  }

  if ("anyOf" in condition) {
    return condition.anyOf.some(expr =>
      evaluateCondition(expr, fieldValues, inputMode)
    );
  }

  if ("not" in condition) {
    return !evaluateCondition(condition.not, fieldValues, inputMode);
  }

  // Handle simple field comparison
  if ("field" in condition) {
    const fieldValue = fieldValues[condition.field];

    // Equality check
    if ("eq" in condition) {
      return fieldValue === condition.eq;
    }

    // Inequality check
    if ("ne" in condition) {
      return fieldValue !== condition.ne;
    }

    // Numeric comparisons
    if (typeof fieldValue === "number") {
      if ("gt" in condition && condition.gt !== undefined) {
        return fieldValue > condition.gt;
      }
      if ("gte" in condition && condition.gte !== undefined) {
        return fieldValue >= condition.gte;
      }
      if ("lt" in condition && condition.lt !== undefined) {
        return fieldValue < condition.lt;
      }
      if ("lte" in condition && condition.lte !== undefined) {
        return fieldValue <= condition.lte;
      }
    }

    // Array membership checks
    if ("in" in condition && condition.in !== undefined) {
      return Array.isArray(condition.in) && condition.in.includes(fieldValue);
    }

    if ("nin" in condition && condition.nin !== undefined) {
      return (
        !Array.isArray(condition.nin) || !condition.nin.includes(fieldValue)
      );
    }
  }

  // Unknown condition type - default to false for safety
  console.warn(
    "[conditionEvaluator] Unknown condition type:",
    condition
  );
  return false;
}

/**
 * Checks if a field should be visible based on its UI metadata.
 *
 * @param uiMetadata - UI metadata from schema property
 * @param fieldValues - Map of field names to their current values
 * @param inputMode - Current input mode
 * @returns true if field should be visible, false otherwise
 */
export function shouldShowField(
  uiMetadata: import("../types").UIMetadata | undefined,
  fieldValues: Record<string, unknown>,
  inputMode?: InputMode
): boolean {
  if (!uiMetadata) {
    return true; // No metadata means always visible
  }

  // Check if hidden
  if (uiMetadata["ui:hidden"] === true) {
    return false;
  }

  // Check if hidden in current mode
  if (
    inputMode &&
    uiMetadata["ui:hideInModes"] &&
    uiMetadata["ui:hideInModes"].includes(inputMode)
  ) {
    return false;
  }

  // Evaluate showIf condition
  if (uiMetadata["ui:showIf"]) {
    return evaluateCondition(
      uiMetadata["ui:showIf"],
      fieldValues,
      inputMode
    );
  }

  return true;
}
