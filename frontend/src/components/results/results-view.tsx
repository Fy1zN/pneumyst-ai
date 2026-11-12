"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, type Analysis } from "@/lib/api";
import { stageName } from "@/lib/format";
import { useApi } from "@/lib/use-api";

import { FindingsPanel } from "./findings-panel";
import { ReportPanel } from "./report-panel";
import { ScanList } from "./scan-list";
import { Viewer } from "./viewer";

type State = { id: string; analysis: Analysis } | { id: string; error: string; notFound: boolean };

// Image links last five minutes; refresh at most this often so a broken image cannot cause a request loop.
const MIN_REFRESH_MS = 30_000;

function PartialBanner({ analysis }: { analysis: Analysis }) {
  const failed = Object.keys(analysis.errors);
  if (analysis.status !== "partial" || failed.length === 0) return null;
  return (
    <p role="status" className="rounded-xl bg-caution px-4 py-3 text-sm text-band-possible">
      <span className="font-semibold">Partial result.</span> These steps did not finish: {failed.map(stageName).join(", ")}. Everything
      shown below did finish and can be trusted as usual.
    </p>
  );
}

/** The results page: the user's scans, the X-ray viewer, findings by band and the report panel. */
export function ResultsView() {
  const { id } = useParams<{ id: string }>();
  const api = useApi();
  const [state, setState] = useState<State | null>(null);
  const [reload, setReload] = useState(0);
  const lastRefresh = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    api<Analysis>(`/analyses/${encodeURIComponent(id)}`, { signal: controller.signal })
      .then((analysis) => setState({ id, analysis }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const notFound = error instanceof ApiError && error.status === 404;
        const message = notFound
          ? "This scan does not exist, or it belongs to another account."
          : error instanceof ApiError
            ? error.message
            : "The result could not be loaded.";
        setState({ id, error: message, notFound });
      });
    return () => controller.abort();
  }, [api, id, reload]);

  const refreshLinks = useCallback(() => {
    const now = Date.now();
    if (now - lastRefresh.current < MIN_REFRESH_MS) return;
    lastRefresh.current = now;
    setReload((n) => n + 1);
  }, []);

  const current = state?.id === id ? state : null;

  return (
    <div className="mx-auto grid w-full max-w-[90rem] flex-1 items-start gap-5 px-4 pt-28 pb-16 md:px-6 lg:grid-cols-[17rem_minmax(0,1fr)] xl:grid-cols-[17rem_minmax(0,1fr)_23rem]">
            <div className="hidden lg:block">
        <ScanList currentId={id} />
      </div>

      {!current ? (
        <p className="label py-24 text-center text-muted xl:col-span-2">Loading the result…</p>
      ) : "error" in current ? (
        <section className="flex flex-col gap-4 rounded-2xl border border-line bg-white p-8 xl:col-span-2">
          <h1 className="font-serif text-3xl text-brand-green italic">{current.notFound ? "Scan not found" : "Could not load this scan"}</h1>
          <p className="text-muted">{current.error}</p>
          <div className="flex flex-wrap gap-3">
            {!current.notFound && (
              <button
                type="button"
                onClick={() => setReload((n) => n + 1)}
                className="label rounded-lg border border-line bg-white px-5 py-3 font-bold text-brand-green-deep hover:bg-brand-mint"
              >
                Try again
              </button>
            )}
            <Link href="/scan" className="label rounded-lg bg-brand-green px-5 py-3 font-bold text-white hover:bg-brand-green-deep">
              Start a new scan
            </Link>
          </div>
        </section>
      ) : (
        <>
          <div className="flex min-w-0 flex-col gap-5">
            <Link href="/history" className="text-sm font-semibold text-brand-green lg:hidden">
              ← All your scans
            </Link>
            <PartialBanner analysis={current.analysis} />
            <Viewer key={current.analysis.id} analysis={current.analysis} onLinksExpired={refreshLinks} />
            <FindingsPanel key={`findings-${current.analysis.id}`} findings={current.analysis.findings} />
          </div>
                   <div className="flex flex-col gap-3 lg:col-start-2 xl:col-start-auto">
            <ReportPanel analysis={current.analysis} />
          </div>
          <p className="label text-[0.5625rem] text-muted lg:col-start-2 xl:col-span-2">
            Scan {current.analysis.id.slice(0, 8).toUpperCase()} · {current.analysis.status === "partial" ? "partial" : "complete"} · saved
            to your account · research prototype, not for clinical use
          </p>
        </>
      )}
    </div>
  );
}