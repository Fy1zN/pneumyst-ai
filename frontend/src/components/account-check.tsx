"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, type AnalysisPage } from "@/lib/api";
import { useApi } from "@/lib/use-api";

type Result = { userId: string; ok: true; total: number } | { userId: string; ok: false; message: string };

/** Temporary F2 check: a signed-in call to the backend proves the token is accepted. Replaced in F5. */
export function AccountCheck() {
  const { isLoaded, isSignedIn, userId } = useAuth();
  const api = useApi();
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    if (!isSignedIn || !userId) return;
    const controller = new AbortController();
    api<AnalysisPage>("/analyses?limit=1", { signal: controller.signal })
      .then((page) => setResult({ userId, ok: true, total: page.total }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message =
          error instanceof ApiError && error.status === 401
            ? "the backend rejected the sign-in token (check AUTH_MODE and the CLERK_ settings)"
            : error instanceof ApiError
              ? error.message
              : "unexpected error";
        setResult({ userId, ok: false, message });
      });
    return () => controller.abort();
  }, [api, isSignedIn, userId]);

  if (!isLoaded) return null;
  if (!isSignedIn) {
    return (
      <p className="text-sm text-muted">
        <Link href="/sign-in" className="font-semibold text-brand-green underline underline-offset-4">
          Sign in
        </Link>{" "}
        to check that the backend accepts your account.
      </p>
    );
  }
  // A result for a previous account is ignored after switching users.
  const current = result?.userId === userId ? result : null;
  if (!current) return <p className="label text-muted">Checking your account…</p>;
  if (!current.ok) {
    return (
      <p className="inline-flex items-center gap-2 rounded-full bg-[#fdecea] px-4 py-2 text-sm text-band-likely">
        <span className="size-2 rounded-full bg-band-likely" aria-hidden="true" />
        Signed in, but {current.message}
      </p>
    );
  }
  return (
    <p className="inline-flex items-center gap-2 rounded-full border border-brand-mint-strong bg-brand-mint px-4 py-2 text-sm text-brand-green-deep">
      <span className="size-2 rounded-full bg-brand-green" aria-hidden="true" />
      <span className="font-semibold">Signed in · token accepted</span>
      <span className="text-muted">
        · {current.total} {current.total === 1 ? "scan" : "scans"} saved
      </span>
    </p>
  );
}