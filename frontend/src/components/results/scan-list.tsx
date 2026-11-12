"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, type AnalysisPage, type AnalysisSummary } from "@/lib/api";
import { when } from "@/lib/format";
import { useApi } from "@/lib/use-api";

const PAGE = 8;

function Counts({ item }: { item: AnalysisSummary }) {
  const likely = item.flagged.filter((f) => f.band === "likely").length;
  const possible = item.flagged.filter((f) => f.band === "possible").length;
  if (!likely && !possible) return <span className="label text-[0.5625rem] text-muted">Clear</span>;
  return (
    <span className="flex items-center gap-2 text-xs font-semibold" aria-label={`${likely} likely, ${possible} possible`}>
      {likely > 0 && (
        <span className="flex items-center gap-1 text-band-likely">
          <span className="size-1.5 rounded-full bg-band-likely" aria-hidden="true" />
          {likely}
        </span>
      )}
      {possible > 0 && (
        <span className="flex items-center gap-1 text-band-possible">
          <span className="size-1.5 rounded-full bg-band-possible" aria-hidden="true" />
          {possible}
        </span>
      )}
    </span>
  );
}

/** The signed-in user's scans, 8 at a time, with the open scan highlighted. */
export function ScanList({ currentId }: { currentId: string }) {
  const api = useApi();
  const [items, setItems] = useState<AnalysisSummary[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api<AnalysisPage>(`/analyses?limit=${PAGE}&offset=0`, { signal: controller.signal })
      .then((page) => {
        setItems(page.items);
        setTotal(page.total);
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(e instanceof ApiError ? e.message : "Your scans could not be loaded.");
      });
    return () => controller.abort();
  }, [api]);

  async function loadMore() {
    setLoadingMore(true);
    try {
      const page = await api<AnalysisPage>(`/analyses?limit=${PAGE}&offset=${items.length}`);
      setItems((current) => [...current, ...page.items.filter((i) => !current.some((c) => c.id === i.id))]);
      setTotal(page.total);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "More scans could not be loaded.");
    } finally {
      setLoadingMore(false);
    }
  }

  const needle = filter.trim().toLowerCase();
  const shown = needle ? items.filter((i) => (i.case_number ?? "").toLowerCase().includes(needle)) : items;

  return (
    <aside aria-label="Your scans" className="flex flex-col gap-4 rounded-2xl border border-line bg-white p-5">
      <p className="label text-brand-green">Your scans</p>
      <label className="flex h-10 items-center gap-2 rounded-lg border border-line bg-page px-3 focus-within:border-brand-green">
        <svg viewBox="0 0 24 24" className="size-3.5 shrink-0" fill="none" stroke="#64748b" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <circle cx="11" cy="11" r="7" />
          <path d="M20 20l-4-4" />
        </svg>
        <span className="sr-only">Filter by case number</span>
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filter by case number"
          className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
        />
      </label>

      {error && <p className="text-sm text-band-likely">{error}</p>}
      {total === null && !error && <p className="label text-muted">Loading…</p>}

      <ul className="flex flex-col gap-1">
        {shown.map((item) => {
          const active = item.id === currentId;
          return (
            <li key={item.id}>
              <Link
                href={`/results/${item.id}`}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 transition ${active ? "border-l-2 border-brand-green bg-brand-mint" : "hover:bg-page"}`}
              >
                <span className="min-w-0 flex-1">
                  <span className={`block truncate text-sm ${item.case_number ? "font-semibold" : "font-medium text-muted"}`}>
                    {item.case_number ?? "No case number"}
                  </span>
                  <span className="block text-xs text-muted">{when(item.created_at)}</span>
                </span>
                <Counts item={item} />
              </Link>
            </li>
          );
        })}
        {total !== null && shown.length === 0 && (
          <li className="px-3 py-2 text-sm text-muted">{needle ? "No loaded scan matches." : "No scans yet."}</li>
        )}
      </ul>

      {total !== null && (
        <div className="mt-auto flex items-center justify-between border-t border-line pt-3">
          <span className="label text-[0.5625rem] text-muted">
            {items.length} of {total} {total === 1 ? "scan" : "scans"}
          </span>
          {items.length < total && (
            <button type="button" onClick={() => void loadMore()} disabled={loadingMore} className="text-sm font-semibold text-brand-green disabled:opacity-50">
              {loadingMore ? "Loading…" : "Load more"}
            </button>
          )}
        </div>
      )}
    </aside>
  );
}