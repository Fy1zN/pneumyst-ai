"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { ApiError, type Analysis } from "@/lib/api";
import { COUGH, DETAIL_FIELDS, RISK_FIELDS, SYMPTOM_FIELDS, optionLabel, type CaseKey } from "@/lib/case-fields";
import { duration, modelName, modelVersion, stageName } from "@/lib/format";
import { useApi } from "@/lib/use-api";

const WHO_KEYS: CaseKey[] = ["cough", "fever", "night_sweats", "weight_loss"];
const SCREEN_STYLE = {
  positive: "bg-caution text-band-possible",
  negative: "bg-page text-band-unlikely",
  incomplete: "bg-page text-muted",
} as const;

function shortHash(hash: string | null | undefined): string {
  return hash ? `${hash.slice(0, 4)}…${hash.slice(-2)}` : "not recorded";
}

function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3 pb-1">
        <h3 className="label text-[0.5625rem] font-bold text-brand-green">{title}</h3>
        {aside}
      </div>
      <dl className="flex flex-col">{children}</dl>
    </section>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2 text-sm last:border-b-0">
      <dt className="text-muted">{label}</dt>
      <dd className="text-right font-medium break-all">{children}</dd>
    </div>
  );
}

function DeleteDialog({ caseName, busy, error, onCancel, onConfirm }: {
  caseName: string;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [busy, onCancel]);

  return (
    <div className="fixed inset-0 z-[60] grid place-items-center bg-ink/30 p-4 backdrop-blur-sm">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="delete-title"
        aria-describedby="delete-text"
        className="flex w-full max-w-[30rem] flex-col gap-4 rounded-[18px] border border-line bg-white p-8 shadow-[0_24px_60px_rgb(10_51_26_/_0.2)]"
      >
              <h2 id="delete-title" className="font-serif text-2xl text-brand-green italic">
          Delete {caseName}?
        </h2>
        <p id="delete-text" className="text-sm text-muted">
          This permanently removes the scores, the X-ray, the heatmaps, the summary and the audit record of this scan from your account. It
          cannot be undone. Download the PDF or the audit record first if you need them.
        </p>
        {error && <p className="text-sm text-band-likely">{error}</p>}
        <div className="flex flex-wrap justify-end gap-3 pt-2">
          <button
            type="button"
            autoFocus
            onClick={onCancel}
            disabled={busy}
            className="label rounded-lg border border-line bg-white px-5 py-3 font-bold text-brand-green-deep hover:bg-brand-mint disabled:opacity-60"
          >
            Keep it
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
                       className="label rounded-lg bg-brand-green px-5 py-3 font-bold text-white transition hover:bg-brand-green-deep disabled:opacity-60"
          >
            {busy ? "Deleting…" : "Delete scan"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Everything recorded about the scan: case context, models, processing, and the option to delete it. */
export function DetailsTab({ analysis }: { analysis: Analysis }) {
  const api = useApi();
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const context = analysis.case_context ?? {};
  const notes = analysis.case_notes;
  const caseName = context.case_number ?? `scan ${analysis.id.slice(0, 8).toUpperCase()}`;
  const answered = (keys: { key: CaseKey; label: string }[]) => keys.filter((f) => context[f.key]);
  const detailRows = answered(DETAIL_FIELDS);
  const otherRows = answered([...SYMPTOM_FIELDS.filter((f) => !WHO_KEYS.includes(f.key)), ...RISK_FIELDS]);
  const whoRows = [COUGH, ...SYMPTOM_FIELDS.filter((f) => WHO_KEYS.includes(f.key))];
  const transform = analysis.preprocess.transform as { orig_w?: number; orig_h?: number } | undefined;
  const failedStages = Object.entries(analysis.stages).filter(([, status]) => status !== "success");
  const timings = Object.entries(analysis.timings_ms).filter(([step]) => step !== "total");

  async function copyId() {
    try {
      await navigator.clipboard.writeText(analysis.id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can be blocked; the id stays visible to copy by hand.
    }
  }

  async function remove() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await api<void>(`/analyses/${analysis.id}`, { method: "DELETE" });
      router.replace("/scan");
    } catch (error) {
      setDeleting(false);
      if (error instanceof ApiError && error.status === 404) {
        router.replace("/scan");
        return;
      }
      setDeleteError(
        error instanceof ApiError && error.code === "NETWORK_ERROR" ? error.message : "The scan could not be deleted. Try again in a moment.",
      );
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <h2 className="font-serif text-2xl text-brand-green italic">Scan details</h2>

      <Section title="Case context">
        <Row label="Case number">{context.case_number ?? "Not entered"}</Row>
        {detailRows.map((f) => (
          <Row key={f.key} label={f.label}>
            {optionLabel(String(context[f.key]))}
          </Row>
        ))}
        {detailRows.length === 0 && <p className="py-2 text-sm text-muted">No other details were entered.</p>}
      </Section>

      {notes?.who_screen && (
        <Section
          title="WHO symptom screen"
          aside={
            <span className={`label rounded-full px-2.5 py-1 text-[0.5625rem] font-bold ${SCREEN_STYLE[notes.who_screen.result]}`}>
              {notes.who_screen.result}
            </span>
          }
        >
          {whoRows.map((f) => {
            const value = context[f.key];
            const text = !value
              ? "Not asked"
              : f.key === "cough" && value === "yes" && context.cough_duration
                ? `Yes, ${context.cough_duration}`
                : optionLabel(String(value));
            return (
              <Row key={f.key} label={f.key === "cough" ? "Current cough" : f.label}>
                <span className={value ? "" : "font-normal text-muted"}>{text}</span>
              </Row>
            );
          })}
        </Section>
      )}

      {otherRows.length > 0 && (
        <Section title="Other symptoms and history">
          {otherRows.map((f) => (
            <Row key={f.key} label={f.label}>
              {optionLabel(String(context[f.key]))}
            </Row>
          ))}
        </Section>
      )}

      {notes && notes.cautions.length > 0 && (
        <div className="flex flex-col gap-2 rounded-xl bg-caution p-4 text-sm">
          <p className="label text-[0.5625rem] font-bold text-band-possible">Caution</p>
          {notes.cautions.map((c) => (
            <p key={c.code}>{c.text}</p>
          ))}
        </div>
      )}
      {analysis.case_context && <p className="-mt-3 text-xs text-muted">Shown next to the result only. Never sent to the models or to the AI.</p>}

      <Section title="Models">
        {analysis.models.map((m) => (
          <Row key={m.name} label={modelName(m.name)}>
            {modelVersion(m.name)}
            <span className="block font-mono text-[0.6875rem] font-normal text-muted">sha256 {shortHash(m.checkpoint_sha256)}</span>
          </Row>
        ))}
      </Section>

      <Section title="Processing">
        <Row label="Uploaded image">
          {analysis.image_format.toUpperCase()}
          {transform?.orig_w && transform.orig_h ? `, ${transform.orig_w} × ${transform.orig_h} px` : ""}
        </Row>
        <Row label="Image fingerprint">
          <span className="font-mono text-xs">sha256 {shortHash(analysis.image_sha256)}</span>
        </Row>
        <Row label="Model input">
          {String(analysis.preprocess.size ?? 320)} × {String(analysis.preprocess.size ?? 320)} px, grayscale
        </Row>
        {timings.map(([step, ms]) => (
          <Row key={step} label={step === "preprocess" ? "Preparing the image" : optionLabel(stageName(step).replace(/^the /, ""))}>
            {duration(ms)}
          </Row>
        ))}
        {analysis.timings_ms.total !== undefined && <Row label="Total screening time">{duration(analysis.timings_ms.total)}</Row>}
        {failedStages.map(([stage, status]) => (
          <Row key={stage} label={optionLabel(stageName(stage).replace(/^the /, ""))}>
            <span className={status === "skipped" ? "text-muted" : "text-band-likely"}>
              {status}
              {analysis.errors[stage] ? ` (${analysis.errors[stage]})` : ""}
            </span>
          </Row>
        ))}
      </Section>

      <Section title="Record">
        <Row label="Scan id">
          <span className="font-mono text-xs">{analysis.id}</span>
          <button type="button" onClick={() => void copyId()} className="mt-1 block w-full text-right text-xs font-semibold text-brand-green">
            {copied ? "Copied" : "Copy id"}
          </button>
        </Row>
        <Row label="Saved">{new Date(analysis.created_at).toLocaleString()}</Row>
      </Section>

      <div className="flex flex-col gap-2 border-t border-line pt-5">
        <button
          type="button"
          onClick={() => {
            setDeleteError(null);
            setConfirming(true);
          }}
                   className="label rounded-lg border border-line bg-white py-3 font-bold text-brand-green-deep transition hover:bg-brand-mint"
        >
          Delete this scan
        </button>
        <p className="text-xs text-muted">Removes this scan and its images from your account for good.</p>
      </div>

      {confirming && (
        <DeleteDialog
          caseName={caseName}
          busy={deleting}
          error={deleteError}
          onCancel={() => setConfirming(false)}
          onConfirm={() => void remove()}
        />
      )}
    </div>
  );
}