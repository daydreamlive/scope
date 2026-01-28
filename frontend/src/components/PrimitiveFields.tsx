/**
 * Primitive field renderers for dynamic settings panel.
 * Exports per-type components (TextField, NumberField, SliderField, ToggleField, EnumField)
 * and SchemaPrimitiveField which dispatches by inferred or explicit fieldType.
 */

import { Input } from "./ui/input";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Toggle } from "./ui/toggle";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { SliderWithInput } from "./ui/slider-with-input";
import { Minus, Plus } from "lucide-react";
import type {
  SchemaProperty,
  SchemaFieldUI,
  PrimitiveFieldType,
} from "../lib/schemaSettings";
import { inferPrimitiveFieldType } from "../lib/schemaSettings";

export interface BaseFieldProps {
  fieldKey: string;
  prop: SchemaProperty;
  value: unknown;
  onChange: (v: unknown) => void;
  disabled?: boolean;
  label?: string;
  tooltip?: string;
}

function formatLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function resolveLabelAndTooltip(
  fieldKey: string,
  description: string | undefined,
  label?: string,
  tooltip?: string
): { label: string; tooltip: string } {
  const resolvedLabel = label ?? description ?? formatLabel(fieldKey);
  return { label: resolvedLabel, tooltip: tooltip ?? description ?? "" };
}

/** Text input field. */
export function TextField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled,
  label,
  tooltip,
}: BaseFieldProps) {
  const { label: displayLabel, tooltip: displayTooltip } =
    resolveLabelAndTooltip(fieldKey, prop.description, label, tooltip);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={displayLabel}
          tooltip={displayTooltip}
          className="text-sm text-foreground"
        />
        <Input
          type="text"
          value={String(value ?? prop.default ?? "")}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
          className="h-8"
        />
      </div>
    </div>
  );
}

/** Number input field (no slider). */
export function NumberField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled,
  label,
  tooltip,
}: BaseFieldProps) {
  const { label: displayLabel, tooltip: displayTooltip } =
    resolveLabelAndTooltip(fieldKey, prop.description, label, tooltip);
  const rawVal = typeof value === "number" ? value : Number(prop.default) || 0;
  const numVal = Math.round(rawVal);
  const min = typeof prop.minimum === "number" ? Math.round(prop.minimum) : 0;
  const max =
    typeof prop.maximum === "number" ? Math.round(prop.maximum) : 2147483647;
  const increment = () => {
    if (numVal >= max) return;
    onChange(numVal + 1);
  };
  const decrement = () => {
    if (numVal <= min) return;
    onChange(numVal - 1);
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseInt(e.target.value, 10);
    if (!Number.isNaN(v)) onChange(v);
  };
  const labelText = `${displayLabel}:`;
  const isLongLabel = labelText.length > 20;
  const stepper = (
    <div
      className={
        isLongLabel
          ? "w-full flex items-center border rounded-full overflow-hidden h-8"
          : "flex-1 min-w-0 flex items-center border rounded-full overflow-hidden h-8"
      }
    >
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
        onClick={decrement}
        disabled={disabled || numVal <= min}
      >
        <Minus className="h-3.5 w-3.5" />
      </Button>
      <Input
        type="number"
        value={numVal}
        onChange={handleInputChange}
        disabled={disabled}
        min={min}
        max={max}
        className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 flex-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
        onClick={increment}
        disabled={disabled || numVal >= max}
      >
        <Plus className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
  return (
    <div className="space-y-2">
      <div
        className={
          isLongLabel ? "flex flex-col gap-1.5" : "flex items-center gap-2"
        }
      >
        <LabelWithTooltip
          label={labelText}
          tooltip={displayTooltip}
          className={
            isLongLabel
              ? "text-sm text-foreground"
              : "text-sm text-foreground w-14 shrink-0"
          }
        />
        {stepper}
      </div>
    </div>
  );
}

/** Slider field (number with min/max). Integer when prop.type !== "number". */
export function SliderField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled,
  label,
  tooltip,
}: BaseFieldProps) {
  const { label: displayLabel, tooltip: displayTooltip } =
    resolveLabelAndTooltip(fieldKey, prop.description, label, tooltip);
  const rawVal = typeof value === "number" ? value : Number(prop.default) || 0;
  const min = typeof prop.minimum === "number" ? prop.minimum : 0;
  const max = typeof prop.maximum === "number" ? prop.maximum : 100;
  const isFloat = prop.type === "number";
  const step = isFloat ? 0.01 : 1;
  const incrementAmount = isFloat ? 0.01 : 1;
  const numVal = isFloat ? rawVal : Math.round(rawVal);
  return (
    <SliderWithInput
      label={displayLabel}
      tooltip={displayTooltip}
      value={numVal}
      onValueChange={v => onChange(isFloat ? v : Math.round(v))}
      onValueCommit={v => onChange(isFloat ? v : Math.round(v))}
      min={min}
      max={max}
      step={step}
      incrementAmount={incrementAmount}
      labelClassName="text-sm text-foreground w-14 shrink-0"
      valueFormatter={isFloat ? (v: number) => v : (v: number) => Math.round(v)}
      inputParser={
        isFloat
          ? (v: string) => parseFloat(v) || numVal
          : (v: string) => Math.round(parseFloat(v) || numVal)
      }
      disabled={disabled}
    />
  );
}

/** Toggle field (boolean). */
export function ToggleField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled,
  label,
  tooltip,
}: BaseFieldProps) {
  const { label: displayLabel, tooltip: displayTooltip } =
    resolveLabelAndTooltip(fieldKey, prop.description, label, tooltip);
  const boolVal = value === true;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={displayLabel}
          tooltip={displayTooltip}
          className="text-sm font-medium"
        />
        <Toggle
          pressed={boolVal}
          onPressedChange={p => onChange(p)}
          variant="outline"
          size="sm"
          className="h-7"
          disabled={disabled}
        >
          {boolVal ? "ON" : "OFF"}
        </Toggle>
      </div>
    </div>
  );
}

export interface EnumFieldProps extends BaseFieldProps {
  /** Enum options from schema $defs when using $ref */
  enumValues?: string[];
}

/** Enum/dropdown field. */
export function EnumField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled,
  label,
  tooltip,
  enumValues,
}: EnumFieldProps) {
  const { label: displayLabel, tooltip: displayTooltip } =
    resolveLabelAndTooltip(fieldKey, prop.description, label, tooltip);
  const options = enumValues ?? (prop.enum as string[]) ?? [];
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label={displayLabel}
          tooltip={displayTooltip}
          className="text-sm text-foreground"
        />
        <Select
          value={String(value ?? prop.default ?? "")}
          onValueChange={v => onChange(v)}
          disabled={disabled}
        >
          <SelectTrigger className="w-[140px] h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map(opt => (
              <SelectItem key={opt} value={opt}>
                {opt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

export interface SchemaPrimitiveFieldProps extends BaseFieldProps {
  ui?: SchemaFieldUI;
  /** When provided, dispatch by this type instead of inferring from prop */
  fieldType?: PrimitiveFieldType;
  /** Enum options from schema $defs when field uses $ref */
  enumValues?: string[];
}

/**
 * Renders a single primitive schema-driven field by dispatching to
 * TextField, NumberField, SliderField, ToggleField, or EnumField.
 * Uses fieldType when given, otherwise infers from prop.
 */
export function SchemaPrimitiveField({
  fieldKey,
  prop,
  value,
  onChange,
  disabled = false,
  label,
  tooltip,
  fieldType,
  enumValues,
}: SchemaPrimitiveFieldProps) {
  const resolvedType: PrimitiveFieldType | null =
    fieldType ?? inferPrimitiveFieldType(prop);
  if (!resolvedType) return null;

  const base = {
    fieldKey,
    prop,
    value,
    onChange,
    disabled,
    label: label ?? (prop.description as string) ?? undefined,
    tooltip: (tooltip ?? prop.description) as string | undefined,
  };

  switch (resolvedType) {
    case "text":
      return <TextField key={fieldKey} {...base} />;
    case "number":
      return <NumberField key={fieldKey} {...base} />;
    case "slider":
      return <SliderField key={fieldKey} {...base} />;
    case "toggle":
      return <ToggleField key={fieldKey} {...base} />;
    case "enum":
      return <EnumField key={fieldKey} {...base} enumValues={enumValues} />;
    default:
      return null;
  }
}
