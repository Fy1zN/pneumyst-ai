import type { components } from "@/lib/api-types";

export type CaseContext = components["schemas"]["CaseContext"];
export type CaseKey = keyof CaseContext;
type ChoiceKey = Exclude<CaseKey, "case_number">;
type Choice<K extends ChoiceKey> = NonNullable<CaseContext[K]>;

/** What the form holds: "" means "Not asked", which is sent as nothing so the backend never reads it as "no". */
export type CaseValues = Partial<Record<CaseKey, string>>;

// The same rule as the backend: letters, digits, '-' and '_' only, so a name with spaces cannot be typed in.
export const CASE_NUMBER_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;

type SelectField<K extends ChoiceKey = ChoiceKey> = { key: K; label: string; options: readonly Choice<K>[] };

function field<K extends ChoiceKey>(key: K, label: string, options: readonly Choice<K>[]): SelectField {
  return { key, label, options } as SelectField;
}

const YES_NO = ["yes", "no"] as const;
const YES_NO_UNKNOWN = ["yes", "no", "unknown"] as const;

// Options are typed against the generated API types, so a choice the backend does not accept fails to compile.
export const DETAIL_FIELDS: SelectField[] = [
  field("age_band", "Age band", ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]),
  field("sex", "Sex", ["female", "male", "other"]),
  field("reason", "Reason for X-ray", [
    "cough or fever", "TB screening", "breathlessness", "chest pain", "follow-up", "pre-operative", "routine check", "other",
  ]),
  field("view", "View", ["PA", "AP", "lateral"]),
  field("position", "Position", ["standing", "sitting", "lying"]),
  field("image_source", "Image source", ["digital X-ray", "scanned film", "phone photo of film"]),
];

export const COUGH = field("cough", "Cough", YES_NO);
export const COUGH_DURATION = field("cough_duration", "Cough duration", ["under 2 weeks", "2 weeks or more"]);

export const SYMPTOM_FIELDS: SelectField[] = [
  field("fever", "Fever", YES_NO),
  field("night_sweats", "Night sweats", YES_NO),
  field("weight_loss", "Weight loss", YES_NO),
  field("breathlessness", "Breathlessness", YES_NO),
  field("chest_pain", "Chest pain", YES_NO),
  field("coughing_blood", "Coughing up blood", YES_NO),
];

export const RISK_FIELDS: SelectField[] = [
  field("tb_contact", "TB contact", YES_NO_UNKNOWN),
  field("previous_tb", "Previous TB", YES_NO_UNKNOWN),
  field("diabetes", "Diabetes", YES_NO_UNKNOWN),
  field("smoking", "Smoking", ["never", "former", "current"]),
];

export type { SelectField };

/** "digital X-ray" -> "Digital X-ray", "TB screening" stays as it is. */
export function optionLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/** Only the answered fields, as the form fields /analyze expects. Cough duration is dropped unless the cough is "yes". */
export function toFormData(values: CaseValues): [string, string][] {
  const entries: [string, string][] = [];
  for (const [key, value] of Object.entries(values)) {
    if (!value) continue;
    if (key === "cough_duration" && values.cough !== "yes") continue;
    entries.push([key, key === "case_number" ? value.trim() : value]);
  }
  return entries;
}