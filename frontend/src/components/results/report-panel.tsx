"use client";

import { useEffect, useState } from "react";

import { ApiError, type Analysis, type Report } from "@/lib/api";
import { useApi, useDownload } from "@/lib/use-api";

type Tab = "report" | "ask" | "details";
type ReportState = { kind: "loading" } | { kind: "ready"; report: Report } | { kind: "error"; message: string };

function CaseContextBox({ analysis }: { analysis: Analysis }) {
  const notes = analysis.case_notes;
  if (!notes || (!notes.who_screen && notes.cautions.length === 0)) return null;
  const screen = notes.who_screen;
  const screenText = !screen
    ? null
    : screen.result === "positive"
      ? `positive (${screen.positive_symptoms.join(", ")})`
      : screen.result === "negative"
        ? "negative (all four answered no)"
        : `incomplete (not asked: ${screen.not_asked.join(", ")})`;
  return (
    <div className="flex flex-col gap-2 rounded-xl bg-caution p-4 text-sm">
      <p className="label text-[0.5625rem] font-bold text-band-possible">Shown next to the result, not used by the model</p>
      {screenText && <p>WHO symptom screen: {screenText}.</p>}
      {notes.cautions.map((c) => (
        <p key={c.code}>Caution: {c.text}</p>
      ))}
    </div>
  );
}

function Writing() {
  return (
    <div className="flex flex-col gap-3" aria-live="polite">
      <p className="flex items-center gap-2.5 rounded-lg bg-brand-mint px-3 py-2.5 text-sm font-semibold text-brand-green-deep">
        <svg viewBox="0 0 20 20" className="size-4 animate-spin motion-reduce:animate-none" aria-hidden="true">
          <circle cx="10" cy="10" r="8" fill="none" stroke="#bbf7d0" strokeWidth="2.5" />
          <path d="M10 2a8 8 0 0 1 8 8" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        Writing the summary
      </p>
      {[100, 94, 98, 90, 62].map((w) => (
        <span key={w} className="h-2.5 animate-pulse rounded-full bg-line motion-reduce:animate-none" style={{ width: `${w}%` }} />
      ))}
      <p className="text-xs text-muted">
        The scores and heatmaps on this page are already final. If writing takes longer than 45 seconds, a plain template summary is shown
        instead.
      </p>
    </div>
  );
}

function ReportTab({ analysis }: { analysis: Analysis }) {
  const api = useApi();
  const download = useDownload();
  const [state, setState] = useState<ReportState>({ kind: "loading" });
  const [attempt, setAttempt] = useState(0);
  const [busy, setBusy] = useState<"pdf" | "audit" | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api<Report>(`/analyses/${analysis.id}/report`, { signal: controller.signal })
      .then((report) => setState({ kind: "ready", report }))
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: e instanceof ApiError ? e.message : "The summary could not be loaded." });
      });
    return () => controller.abort();
  }, [api, analysis.id, attempt]);

  const name = analysis.case_context?.case_number ?? analysis.id.slice(0, 8);
  async function save(kind: "pdf" | "audit") {
    setBusy(kind);
    setDownloadError(null);
    try {
      if (kind === "pdf") await download(`/analyses/${analysis.id}/report.pdf?download=true`, `pneumyst-${name}-report.pdf`);
      else await download(`/analyses/${analysis.id}/audit.json?download=true`, `pneumyst-${name}-audit.json`);
    } catch (e) {
      setDownloadError(e instanceof ApiError ? `The download failed: ${e.message}` : "The download failed.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex flex-1 flex-col gap-4">
      <h2 className="font-serif text-2xl text-brand-green italic">Screening summary</h2>
      {state.kind === "loading" && <Writing />}
      {state.kind === "error" && (
        <div className="flex flex-col items-start gap-2 text-sm">
          <p className="text-band-likely">{state.message}</p>
          <button
            type="button"
            onClick={() => {
              setState({ kind: "loading" });
              setAttempt((n) => n + 1);
            }}
            className="font-semibold text-brand-green underline underline-offset-4"
          >
            Try again
          </button>
        </div>
      )}
      {state.kind === "ready" && (
        <>
          <p className="flex items-start gap-2 rounded-md bg-brand-mint px-3 py-2 text-xs text-brand-green-deep">
            <span className="mt-1 size-1.5 shrink-0 rounded-full bg-brand-green" aria-hidden="true" />
            {state.report.source === "llm"
              ? `Written by ${state.report.model ?? "the language model"}, checked against the scores by fixed rules.`
              : "Written from a fixed template; the language model was not used for this summary."}
          </p>
          <div className="flex flex-col gap-3 text-[0.9375rem] leading-relaxed">
            {state.report.summary
              .split(/\n\s*\n/)
              .filter(Boolean)
              .map((paragraph, i) => (
                <p key={i}>{paragraph}</p>
              ))}
          </div>
          <p className="text-xs text-muted">{state.report.disclaimer}</p>
        </>
      )}
      <CaseContextBox analysis={analysis} />

      <div className="mt-auto flex flex-col gap-2.5 pt-2">
        <button
          type="button"
          onClick={() => void save("pdf")}
          disabled={busy !== null}
          className="label rounded-lg bg-brand-green py-3.5 font-bold text-white transition hover:bg-brand-green-deep disabled:opacity-60"
        >
          {busy === "pdf" ? "Preparing the PDF…" : "Download PDF report"}
        </button>
        <button
          type="button"
          onClick={() => void save("audit")}
          disabled={busy !== null}
          className="label rounded-lg border border-line bg-white py-3.5 font-bold text-brand-green-deep transition hover:bg-brand-mint disabled:opacity-60"
        >
          {busy === "audit" ? "Preparing…" : "Download audit record"}
        </button>
        {downloadError && <p className="text-sm text-band-likely">{downloadError}</p>}
        <p className="text-xs text-muted">Screening aid only, not a diagnosis.</p>
      </div>
    </div>
  );
}

/** Right-hand panel with the Report, Ask and Details tabs. Ask and Details are built in step F4b. */
export function ReportPanel({ analysis }: { analysis: Analysis }) {
  const [tab, setTab] = useState<Tab>("report");
  return (
    <aside aria-label="Report" className="flex flex-col gap-5 rounded-2xl border border-line bg-white p-5">
      <div role="tablist" aria-label="Result tabs" className="flex gap-5 border-b border-line">
        {(["report", "ask", "details"] as const).map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`label -mb-px border-b-2 pt-1 pb-2.5 text-[0.625rem] ${tab === t ? "border-brand-green font-bold text-brand-green-deep" : "border-transparent text-muted"}`}
          >
            {t}
          </button>
        ))}
      </div>
      {tab === "report" ? (
        <ReportTab key={analysis.id} analysis={analysis} />
      ) : (
        <p className="text-sm text-muted">The {tab === "ask" ? "Ask" : "Details"} tab is built in the next step.</p>
      )}
    </aside>
  );
}