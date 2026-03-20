/**
 * Pure math evaluation shared by MathNode (rendered) and the headless subgraph evaluator.
 */
export function computeResult(
  op: string,
  a: number | null,
  b: number | null
): number | null {
  if (a === null) return null;

  // Unary ops
  switch (op) {
    case "abs":
      return Math.abs(a);
    case "negate":
      return -a;
    case "sqrt":
      return a >= 0 ? Math.sqrt(a) : null;
    case "floor":
      return Math.floor(a);
    case "ceil":
      return Math.ceil(a);
    case "round":
      return Math.round(a);
    case "toInt":
      return Math.trunc(a);
    case "toFloat":
      return a + 0.0;
  }

  // Binary ops
  if (b === null) return null;

  switch (op) {
    case "add":
      return a + b;
    case "subtract":
      return a - b;
    case "multiply":
      return a * b;
    case "divide":
      return b !== 0 ? a / b : null;
    case "mod":
      return b !== 0 ? a % b : null;
    case "min":
      return Math.min(a, b);
    case "max":
      return Math.max(a, b);
    case "power":
      return Math.pow(a, b);
    default:
      return null;
  }
}
