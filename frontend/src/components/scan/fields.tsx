import { optionLabel, type SelectField } from "@/lib/case-fields";

const box =
  "h-11 w-full rounded-lg border bg-white px-3.5 text-sm font-medium text-ink transition outline-none focus:border-brand-green focus:ring-3 focus:ring-brand-green/20";

export function FieldLabel({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="label text-[0.625rem] text-brand-green">
      {children}
    </label>
  );
}

/** A dropdown whose first choice, "Not asked", sends nothing. */
export function SelectInput({
  field,
  value,
  onChange,
  disabled,
}: {
  field: SelectField;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const id = `case-${field.key}`;
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <FieldLabel htmlFor={id}>{field.label}</FieldLabel>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className={`${box} border-line ${value ? "" : "text-muted"}`}
      >
        <option value="">Not asked</option>
        {field.options.map((option) => (
          <option key={option} value={option}>
            {optionLabel(option)}
          </option>
        ))}
      </select>
    </div>
  );
}

export function CaseNumberInput({
  value,
  error,
  onChange,
  disabled,
}: {
  value: string;
  error: string | null;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <FieldLabel htmlFor="case-case_number">Case number</FieldLabel>
      <input
        id="case-case_number"
        value={value}
        disabled={disabled}
        maxLength={32}
        autoComplete="off"
        spellCheck={false}
        placeholder="e.g. CASE-0042"
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? "case-number-error" : undefined}
        onChange={(event) => onChange(event.target.value)}
        className={`${box} placeholder:font-normal placeholder:text-muted ${error ? "border-band-likely" : "border-line"}`}
      />
      {error && (
        <p id="case-number-error" className="text-xs text-band-likely">
          {error}
        </p>
      )}
    </div>
  );
}