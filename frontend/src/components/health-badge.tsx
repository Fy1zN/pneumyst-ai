"use client";

import { useEffect, useState } from "react";

import { ApiError, getHealth, type Health } from "@/lib/api";

type State = { kind: "loading" } | { kind: "ok"; health: Health } | { kind: "error"; message: string };

const DOT: Record<Health["status"], string> = {
  ok: "bg-brand-green",
  degraded: "bg-band-possible",
  down: "bg-band-likely",
};

/** Temporary check that the website can reach the backend. The real landing page replaces it in F5. */
export function HealthBadge() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((health) => setState({ kind: "ok", health }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({ kind: "error", message: error instanceof ApiError ? error.message : "Unexpected error" });
      });
    return () => controller.abort();
  }, []);

  if (state.kind === "loading") {
    return <p className="label text-muted">Checking the backend…</p>;
  }
  if (state.kind === "error") {
    return (
      <p className="inline-flex items-center gap-2 rounded-full bg-[#fdecea] px-4 py-2 text-sm text-band-likely">
        <span className="size-2 rounded-full bg-band-likely" aria-hidden="true" />
        Backend not reachable: {state.message}
      </p>
    );
  }
  const { health } = state;
  return (
    <p className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 rounded-full border border-brand-mint-strong bg-brand-mint px-4 py-2 text-sm text-brand-green-deep">
      <span className={`size-2 rounded-full ${DOT[health.status]}`} aria-hidden="true" />
      <span className="font-semibold">Backend: {health.status}</span>
      <span className="label text-[0.625rem] text-muted">
        {health.device} · v{health.version} · database {health.database} · llm {health.llm}
      </span>
    </p>
  );
}