"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import type { Finding } from "@/lib/api";

function Dialog({ labelledBy, children }: { labelledBy: string; children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-[60] grid place-items-center bg-ink/30 p-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className="w-full max-w-[32.5rem] rounded-[18px] border border-line bg-white px-8 pt-9 pb-8 shadow-[0_24px_60px_rgb(10_51_26_/_0.2)]"
      >
        {children}
      </div>
    </div>
  );
}

// Typical durations on the reference GPU; the backend answers once, so the steps only estimate progress.
const STEPS = [
  { label: "Checking the image", ms: 300 },
  { label: "Tuberculosis model", ms: 1200 },
  { label: "14-finding model", ms: 1500 },
  { label: "Heatmaps for flagged findings", ms: 7000, heatmaps: true },
  { label: "Saving to your history", ms: 600 },
];

type StepState = "done" | "running" | "waiting";

function stepStates(durations: number[], elapsed: number, finished: boolean): StepState[] {
  let start = 0;
  return durations.map((ms, index) => {
    const end = start + ms;
    const state: StepState = finished
      ? "done"
      : index === durations.length - 1
        ? // The last step stays "running" until the real answer arrives, however long that takes.
          elapsed >= start
          ? "running"
          : "waiting"
        : elapsed >= end
          ? "done"
          : elapsed >= start
            ? "running"
            : "waiting";
    start = end;
    return state;
  });
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "done") {
    return (
      <svg viewBox="0 0 20 20" className="size-5 shrink-0" aria-hidden="true">
        <circle cx="10" cy="10" r="10" fill="#16a34a" />
        <path d="M6 10.5l2.6 2.6L14 7.6" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (state === "running") {
    return (
      <svg viewBox="0 0 20 20" className="size-5 shrink-0 animate-spin motion-reduce:animate-none" aria-hidden="true">
        <circle cx="10" cy="10" r="8" fill="none" stroke="#bbf7d0" strokeWidth="2.5" />
        <path d="M10 2a8 8 0 0 1 8 8" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 20" className="size-5 shrink-0" aria-hidden="true">
      <circle cx="10" cy="10" r="8" fill="none" stroke="#e2e8f0" strokeWidth="2" />
    </svg>
  );
}

/** "Reading the X-ray" card from docs/design/screens/screening.png. */
export function ProgressDialog({ caseNumber, heatmaps, finished }: { caseNumber: string; heatmaps: boolean; finished: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Date.now() - started), 150);
    return () => window.clearInterval(timer);
  }, []);

  const steps = STEPS.filter((step) => heatmaps || !step.heatmaps);
  const states = stepStates(steps.map((step) => step.ms), elapsed, finished);
  const total = steps.reduce((sum, step) => sum + step.ms, 0);
  const percent = finished ? 100 : Math.min(92, (elapsed / total) * 100);

  return (
    <Dialog labelledBy="progress-title">
      <p className="label text-brand-green">Screening</p>
      <h2 id="progress-title" className="mt-3 font-serif text-3xl text-brand-green-deep italic">
        Reading the X-ray
      </h2>
      <p className="mt-3 text-sm text-muted">
        {caseNumber ? `${caseNumber}. ` : ""}This usually takes {heatmaps ? "5 to 20" : "a few"} seconds. Keep this tab open.
      </p>
      <div
        className="mt-5 h-1.5 overflow-hidden rounded-full bg-line"
        role="progressbar"
        aria-label="Screening progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
      >
        <div className="h-full rounded-full bg-brand-green transition-[width] duration-300" style={{ width: `${percent}%` }} />
      </div>
      <ol className="mt-4 flex flex-col gap-1" aria-live="polite">
        {steps.map((step, index) => {
          const state = states[index];
          return (
            <li
              key={step.label}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 ${state === "running" ? "bg-brand-mint" : ""}`}
            >
              <StepIcon state={state} />
              <span
                className={`flex-1 text-sm ${state === "running" ? "font-semibold" : "font-medium"} ${state === "waiting" ? "text-muted" : "text-ink"}`}
              >
                {step.label}
              </span>
              {state !== "waiting" && (
                <span className={`label text-[0.5625rem] ${state === "running" ? "font-bold text-brand-green" : "text-muted"}`}>
                  {state === "running" ? "Running" : "Done"}
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <p className="mt-5 border-l-[3px] border-brand-green pl-3.5 text-xs leading-relaxed text-brand-green-deep">
        Only the image goes to the models. Your case details are saved next to the result and never change a score.
      </p>
    </Dialog>
  );
}

function AlertIcon() {
  return (
    <span className="grid size-10 place-items-center rounded-full bg-[#fdecea] text-lg font-bold text-band-likely" aria-hidden="true">
      !
    </span>
  );
}

const primary = "label rounded-lg bg-brand-green px-5 py-3 font-bold text-white transition hover:bg-brand-green-deep";
const secondary = "label rounded-lg border border-line bg-white px-5 py-3 font-bold text-brand-green-deep transition hover:bg-brand-mint";

export function FailedDialog({
  message,
  reference,
  code,
  onRetry,
  onClose,
}: {
  message: string;
  reference: string | null;
  code: string;
  onRetry: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog labelledBy="failed-title">
      <div className="flex flex-col items-center gap-3 text-center">
        <AlertIcon />
        <h2 id="failed-title" className="font-serif text-[1.625rem] text-brand-green-deep italic">
          Screening could not run
        </h2>
        <p className="text-sm text-muted">{message} Nothing was saved. Wait a minute and try again.</p>
        <div className="mt-2 flex flex-wrap justify-center gap-2.5">
          <button type="button" className={primary} onClick={onRetry}>
            Try again
          </button>
          <button type="button" className={secondary} onClick={onClose}>
            Back to the form
          </button>
        </div>
        <p className="label text-[0.5625rem] text-muted">
          {reference ? `Ref ${reference} · ` : ""}
          {code}
        </p>
      </div>
    </Dialog>
  );
}

export function SessionDialog() {
  return (
    <Dialog labelledBy="session-title">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="grid size-10 place-items-center rounded-full bg-brand-mint" aria-hidden="true">
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="#16a34a" strokeWidth="2">
            <rect x="6" y="10.5" width="12" height="9" rx="2" />
            <path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" />
          </svg>
        </span>
        <h2 id="session-title" className="font-serif text-[1.625rem] text-brand-green-deep italic">
          Your session has ended
        </h2>
        <p className="text-sm text-muted">
          Sign in again to carry on. Your image was not uploaded, so you will need to add it again.
        </p>
        <Link href="/sign-in?redirect_url=%2Fscan" className={`${primary} mt-2`}>
          Sign in
        </Link>
      </div>
    </Dialog>
  );
}

/** The scan ran but could not be saved: show what was flagged now, because it will not be in the history. */
export function NotSavedDialog({ findings, onClose }: { findings: Finding[]; onClose: () => void }) {
  const flagged = findings.filter((f) => f.band === "likely" || f.band === "possible");
  return (
    <Dialog labelledBy="notsaved-title">
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <AlertIcon />
          <h2 id="notsaved-title" className="font-serif text-[1.625rem] text-brand-green-deep italic">
            This result was not saved
          </h2>
        </div>
        <p className="text-sm text-muted">
          It will not appear in your history, and the PDF and audit downloads need a saved result. Run the scan again to keep
          it.
        </p>
        <div className="rounded-xl bg-caution px-4 py-3 text-sm">
          {flagged.length ? (
            <ul className="flex flex-col gap-1">
              {flagged.map((f) => (
                <li key={f.finding}>
                  <span className="font-semibold">{f.display_name}</span>: {f.band}
                </li>
              ))}
            </ul>
          ) : (
            <p>Nothing was flagged as possible or likely.</p>
          )}
        </div>
        <button type="button" className={`${secondary} self-start`} onClick={onClose}>
          Back to the form
        </button>
      </div>
    </Dialog>
  );
}