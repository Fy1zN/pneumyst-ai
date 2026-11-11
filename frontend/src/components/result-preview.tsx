"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, type Analysis } from "@/lib/api";
import { optionLabel } from "@/lib/case-fields";
import { useApi } from "@/lib/use-api";

type State = { id: string; analysis: Analysis } | { id: string; error: string };

const BAND_STYLE: Record<string, string> = {
  likely: "bg-[#fdecea] text-band-likely",
  possible: "bg-caution text-band-possible",
};

function percent(score: number): string {
  const value = score * 100;
  return value >= 10 ? `${value.toFixed(0)}%` : value >= 1 ? `${value.toFixed(1)}%` : `${value.toFixed(2)}%`;
}

/** Temporary F3 view of a saved scan: case number, time and flagged findings. Replaced by the full results page in F4. */
export function ResultPreview() {
  const { id } = useParams<{ id: string }>();
  const api = useApi();
  const [state, setState] = useState<State | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api<Analysis>(`/analyses/${encodeURIComponent(id)}`, { signal: controller.signal })
      .then((analysis) => setState({ id, analysis }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message =
          error instanceof ApiError && error.status === 404
            ? "This scan does not exist, or it belongs to another account."
            : error instanceof ApiError
              ? error.message
              : "The result could not be loaded.";
        setState({ id, error: message });
      });
    return () => controller.abort();
  }, [api, id]);

  const current = state?.id === id ? state : null;
  if (!current) return <p className="label text-muted">Loading the result…</p>;
  if ("error" in current) {
    return (
      <section className="flex flex-col gap-4 rounded-[18px] border border-line bg-white p-8">
        <h1 className="font-serif text-3xl text-brand-green italic">Scan not found</h1>
        <p className="text-muted">{current.error}</p>
        <Link href="/scan" className="label w-fit rounded-lg bg-brand-green px-5 py-3 font-bold text-white">
          Start a new scan
        </Link>
      </section>
    );
  }

  const { analysis } = current;
  const flagged = analysis.findings
    .filter((f) => f.band === "likely" || f.band === "possible")
    .sort((a, b) => (a.band === b.band ? (b.score ?? 0) - (a.score ?? 0) : a.band === "likely" ? -1 : 1));
  const caseNumber = analysis.case_context?.case_number;
  const when = new Date(analysis.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  return (
    <section className="flex flex-col gap-5 rounded-[18px] border border-line bg-white p-8">
      <div>
        <p className="label text-brand-green">Result saved</p>
        <h1 className="mt-2 font-serif text-4xl text-brand-green italic">{caseNumber ?? "No case number"}</h1>
        <p className="mt-1 text-sm text-muted">
          {when} · {analysis.status === "partial" ? "Some parts did not run" : "Complete"}
        </p>
      </div>
      <div>
        <p className="label mb-2 text-[0.625rem] text-muted">Flagged findings</p>
        {flagged.length ? (
          <ul className="flex flex-wrap gap-2">
            {flagged.map((f) => (
              <li key={f.finding} className={`rounded-full px-3 py-1.5 text-sm font-medium ${BAND_STYLE[f.band ?? ""]}`}>
                {f.display_name} · {f.band} · {f.score === null ? "no score" : percent(f.score)}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted">Nothing was flagged as possible or likely.</p>
        )}
      </div>
      {analysis.case_context && (
        <p className="text-sm text-muted">
          Case details saved:{" "}
          {Object.entries(analysis.case_context)
            .filter(([key, value]) => value && key !== "case_number")
            .map(([key, value]) => `${optionLabel(key.replaceAll("_", " ")).replace(/\btb\b/, "TB")}: ${value}`)
            .join(" · ") || "only the case number"}
        </p>
      )}
      <p className="rounded-xl bg-brand-mint px-4 py-3 text-sm text-brand-green-deep">
        The full results page with the X-ray viewer, heatmaps and report comes in the next step.
      </p>
      <Link href="/scan" className="label w-fit rounded-lg border border-line px-5 py-3 font-bold text-brand-green-deep hover:bg-brand-mint">
        Start another scan
      </Link>
    </section>
  );
}