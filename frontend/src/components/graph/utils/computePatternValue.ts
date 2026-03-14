/**
 * Compute an animated value from a control-node pattern.
 * Extracted so it can be reused by the parent-value bridge.
 */
export function computePatternValue(
  pattern: "sine" | "bounce" | "random_walk" | "linear" | "step",
  t: number,
  speed: number,
  min: number,
  max: number,
  lastValue: number
): number {
  const range = max - min;
  const phase = (t * speed) % 1;

  switch (pattern) {
    case "sine":
      return min + range * (0.5 + 0.5 * Math.sin(phase * 2 * Math.PI));
    case "bounce": {
      const triangle = phase < 0.5 ? phase * 2 : 2 - phase * 2;
      return min + range * triangle;
    }
    case "random_walk": {
      const step = (Math.random() - 0.5) * 0.1 * range;
      const newValue = lastValue + step;
      return Math.max(min, Math.min(max, newValue));
    }
    case "linear":
      return min + range * phase;
    case "step": {
      const steps = 10;
      const stepIndex = Math.floor(phase * steps);
      return min + (range * stepIndex) / (steps - 1);
    }
    default:
      return min;
  }
}
