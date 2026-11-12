import type { Finding } from "@/lib/api";

/** Scores as short percentages, the same rule as the backend's report: small values keep enough digits. */
export function percent(score: number): string {
  const value = score * 100;
  if (value >= 10) return `${value.toFixed(0)}%`;
  return value >= 1 ? `${value.toFixed(1)}%` : `${value.toFixed(2)}%`;
}

export function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export type Band = "likely" | "possible" | "unlikely";

export const BAND_TEXT: Record<Band, string> = {
  likely: "text-band-likely",
  possible: "text-band-possible",
  unlikely: "text-band-unlikely",
};
export const BAND_DOT: Record<Band, string> = {
  likely: "bg-band-likely",
  possible: "bg-band-possible",
  unlikely: "bg-band-unlikely",
};

/** The 15 findings split the way the page shows them; nothing is dropped. */
export function groupFindings(findings: Finding[]) {
  const byScore = (a: Finding, b: Finding) => (b.score ?? 0) - (a.score ?? 0);
  const scored = findings.filter((f) => f.status === "success" && f.score !== null);
  return {
    likely: scored.filter((f) => f.band === "likely").sort(byScore),
    possible: scored.filter((f) => f.band === "possible").sort(byScore),
    unlikely: scored.filter((f) => f.band === "unlikely").sort(byScore),
    notBanded: scored.filter((f) => f.band === null),
    unavailable: findings.filter((f) => f.status !== "success" || f.score === null),
  };
}

/** Backend model run names, e.g. "tb_densenet121_v2", in plain words. */
export function modelName(name: string): string {
  if (name.startsWith("tb")) return "Tuberculosis model";
  if (name.startsWith("nih")) return "14-finding model";
  return name;
}

/** "tb_densenet121_v2" -> "DenseNet-121, v2". */
export function modelVersion(name: string): string {
  const match = /densenet(\d+)_v(\d+)/i.exec(name);
  return match ? `DenseNet-${match[1]}, v${match[2]}` : name;
}

/** Pipeline stage names from the backend ("input", a model run name, "explain:<run>", "images"), as people would say them. */
export function stageName(stage: string): string {
  if (stage === "input") return "checking the image";
  if (stage === "images") return "saving the images";
  if (stage.startsWith("explain:")) return `heatmaps for the ${modelName(stage.slice(8)).toLowerCase()}`;
  return `the ${modelName(stage).toLowerCase()}`;
}

export function duration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}
