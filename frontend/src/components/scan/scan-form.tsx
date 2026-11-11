"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";

import { CaseNumberInput, SelectInput } from "@/components/scan/fields";
import { FailedDialog, NotSavedDialog, ProgressDialog, SessionDialog } from "@/components/scan/scan-dialogs";
import { UploadBox } from "@/components/scan/upload-box";
import type { Analysis, Finding } from "@/lib/api";
import {
  CASE_NUMBER_PATTERN,
  COUGH,
  COUGH_DURATION,
  DETAIL_FIELDS,
  RISK_FIELDS,
  SYMPTOM_FIELDS,
  toFormData,
  type CaseValues,
} from "@/lib/case-fields";
import { checkFile, describe } from "@/lib/scan-errors";
import { useApi } from "@/lib/use-api";

type Phase =
  | { name: "idle" }
  | { name: "running"; finished: boolean }
  | { name: "failed"; message: string; reference: string | null; code: string }
  | { name: "session" }
  | { name: "notSaved"; findings: Finding[] };

type Notice = { kind: "limit"; minutes: number | null } | { kind: "offline" } | null;

const CASE_NUMBER_HINT = "Use letters, digits, - or _ only, up to 32 characters. Never a name.";
const panel = "rounded-[18px] border border-white/70 bg-white/55 p-6 shadow-[0_12px_32px_rgb(20_83_45_/_0.06)] backdrop-blur-[6px] md:p-9";

function HeatmapToggle({ on, onChange, disabled }: { on: boolean; onChange: (on: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className="flex items-center gap-3 text-sm font-medium text-ink disabled:opacity-60"
    >
      <span className={`flex h-6 w-10 items-center rounded-full p-0.5 transition ${on ? "bg-brand-green" : "bg-line"}`}>
        <span className={`size-5 rounded-full bg-white shadow transition ${on ? "translate-x-4" : ""}`} />
      </span>
      Generate heatmaps
    </button>
  );
}

function NoticeBanner({ notice }: { notice: Notice }) {
  if (!notice) return null;
  const limit = notice.kind === "limit";
  return (
    <div
      role="alert"
      className={`flex gap-3.5 rounded-xl border-l-[3px] px-4 py-4 ${limit ? "border-band-possible bg-caution" : "border-band-likely bg-[#fdecea]"}`}
    >
      <span
        className={`grid size-5 shrink-0 place-items-center rounded-full text-xs font-bold text-white ${limit ? "bg-band-possible" : "bg-band-likely"}`}
        aria-hidden="true"
      >
        !
      </span>
      <div className="text-sm">
        <p className="font-semibold">{limit ? "You have reached this hour’s limit" : "Cannot reach PNEUMYST"}</p>
        <p className="mt-1 text-ink/80">
          {limit
            ? `You can run 30 scans an hour.${notice.minutes ? ` You can start the next one in ${notice.minutes} ${notice.minutes === 1 ? "minute" : "minutes"}.` : " Try again a little later."}`
            : "Check your internet connection. Nothing you entered has been lost."}
        </p>
      </div>
    </div>
  );
}

export function ScanForm() {
  const api = useApi();
  const router = useRouter();
  const [values, setValues] = useState<CaseValues>({});
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [caseNumberError, setCaseNumberError] = useState<string | null>(null);
  const [heatmaps, setHeatmaps] = useState(true);
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const [notice, setNotice] = useState<Notice>(null);
  const busy = phase.name === "running";

  // Ask before closing the tab while a scan is running, so it is not lost halfway.
  useEffect(() => {
    if (!busy) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [busy]);

  const set = (key: keyof CaseValues) => (value: string) => setValues((current) => ({ ...current, [key]: value }));

  const chooseFile = (next: File | null) => {
    setFile(next);
    setFileError(next ? checkFile(next) : null);
  };

  const setCaseNumber = (value: string) => {
    set("case_number")(value);
    setCaseNumberError(value.trim() && !CASE_NUMBER_PATTERN.test(value.trim()) ? CASE_NUMBER_HINT : null);
  };

  async function run() {
    if (busy) return;
    const caseNumber = (values.case_number ?? "").trim();
    if (caseNumber && !CASE_NUMBER_PATTERN.test(caseNumber)) {
      setCaseNumberError(CASE_NUMBER_HINT);
      return;
    }
    if (!file) {
      setFileError("Add a chest X-ray first.");
      return;
    }
    const problem = checkFile(file);
    if (problem) {
      setFileError(problem);
      return;
    }

    const body = new FormData();
    body.append("file", file);
    for (const [key, value] of toFormData(values)) body.append(key, value);

    setNotice(null);
    setPhase({ name: "running", finished: false });
    try {
      const result = await api<Analysis>(`/analyze?explain=${heatmaps}`, { method: "POST", body });
      if (result.persistence === "not_saved") {
        setPhase({ name: "notSaved", findings: result.findings });
        return;
      }
      setPhase({ name: "running", finished: true });
      router.push(`/results/${result.id}`);
    } catch (error) {
      const found = describe(error);
      setPhase({ name: "idle" });
      if (found.kind === "image") setFileError(found.message);
      else if (found.kind === "field" && found.field === "case_number") setCaseNumberError(found.message);
      else if (found.kind === "limit") setNotice({ kind: "limit", minutes: found.minutes });
      else if (found.kind === "offline") setNotice({ kind: "offline" });
      else if (found.kind === "session") setPhase({ name: "session" });
      else if (found.kind === "failed") setPhase({ name: "failed", message: found.message, reference: found.reference, code: found.code });
      else setPhase({ name: "failed", message: found.message, reference: null, code: "INVALID_REQUEST" });
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void run();
  };

  const canRun = !busy && !caseNumberError && !fileError;
  const runButton = (
    <button
      type="submit"
      disabled={!canRun}
      className="label w-full rounded-lg bg-brand-green py-4 font-bold text-white transition hover:bg-brand-green-deep disabled:cursor-not-allowed disabled:opacity-50"
    >
      {busy ? "Screening…" : "Run screening"}
    </button>
  );

  return (
    <form onSubmit={submit} noValidate className="relative z-10 mx-auto flex w-full max-w-[66rem] flex-col gap-6 px-4 pt-28 pb-36 md:pb-20">
      <NoticeBanner notice={notice} />

      <div className="grid items-stretch gap-6 md:grid-cols-2 md:gap-8">
        <section aria-labelledby="case-title" className={`${panel} order-2 md:order-1`}>
          <h1 id="case-title" className="font-serif text-3xl text-brand-green italic">
            New Scan
          </h1>
          <p className="mt-2 text-sm text-muted">Enter the case details and upload the chest X-ray. Every detail is optional.</p>

          <fieldset disabled={busy} className="mt-6 grid grid-cols-2 gap-x-4 gap-y-5">
            <legend className="sr-only">Case details</legend>
            <CaseNumberInput value={values.case_number ?? ""} error={caseNumberError} onChange={setCaseNumber} />
            {DETAIL_FIELDS.map((field) => (
              <div key={field.key} className={field.key === "image_source" ? "col-span-2" : ""}>
                <SelectInput field={field} value={values[field.key] ?? ""} onChange={set(field.key)} />
              </div>
            ))}

            <p className="label col-span-2 mt-2 border-t border-line pt-5 text-[0.625rem] text-brand-green-deep">
              Symptoms · shown next to the result, never used by the model
            </p>
            <SelectInput field={COUGH} value={values.cough ?? ""} onChange={set("cough")} />
            {values.cough === "yes" ? (
              <SelectInput field={COUGH_DURATION} value={values.cough_duration ?? ""} onChange={set("cough_duration")} />
            ) : (
              <span aria-hidden="true" />
            )}
            {SYMPTOM_FIELDS.map((field) => (
              <SelectInput key={field.key} field={field} value={values[field.key] ?? ""} onChange={set(field.key)} />
            ))}

            <p className="label col-span-2 mt-2 border-t border-line pt-5 text-[0.625rem] text-brand-green-deep">TB risk factors</p>
            {RISK_FIELDS.map((field) => (
              <SelectInput key={field.key} field={field} value={values[field.key] ?? ""} onChange={set(field.key)} />
            ))}
          </fieldset>

          <div className="mt-8 hidden flex-col items-center gap-2.5 md:flex">
            {runButton}
            <p className="text-xs text-muted">Screening aid only, not a diagnosis.</p>
          </div>
        </section>

        <section aria-labelledby="upload-title" className={`${panel} order-1 flex flex-col md:order-2`}>
          <h2 id="upload-title" className="font-serif text-3xl text-brand-green italic">
            Image Upload
          </h2>
          <p className="mt-2 mb-6 text-sm text-muted">One frontal chest X-ray per scan.</p>
          <UploadBox file={file} error={fileError} disabled={busy} onFile={chooseFile} />
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <span className="hidden md:inline-flex">
              <HeatmapToggle on={heatmaps} onChange={setHeatmaps} disabled={busy} />
            </span>
            <p className="text-xs text-muted">Never enter names. Use a case number.</p>
          </div>
        </section>
      </div>

      {/* On phones the button stays at the bottom of the screen. */}
      <div className="fixed inset-x-0 bottom-0 z-40 flex flex-col gap-3 border-t border-line bg-white/95 px-4 pt-3.5 pb-6 backdrop-blur md:hidden">
        <HeatmapToggle on={heatmaps} onChange={setHeatmaps} disabled={busy} />
        {runButton}
      </div>

      {phase.name === "running" && (
        <ProgressDialog caseNumber={(values.case_number ?? "").trim()} heatmaps={heatmaps} finished={phase.finished} />
      )}
      {phase.name === "failed" && (
        <FailedDialog
          message={phase.message}
          reference={phase.reference}
          code={phase.code}
          onRetry={() => void run()}
          onClose={() => setPhase({ name: "idle" })}
        />
      )}
      {phase.name === "session" && <SessionDialog />}
      {phase.name === "notSaved" && <NotSavedDialog findings={phase.findings} onClose={() => setPhase({ name: "idle" })} />}
    </form>
  );
}