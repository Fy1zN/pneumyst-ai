import type { Metadata } from "next";
import { Suspense } from "react";

import { ResultPreview } from "@/components/result-preview";

export const metadata: Metadata = { title: "Result" };

// Temporary result page for F3: proves the scan was saved to the signed-in account. The full results page comes in F4.
export default function ResultPage() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 pt-32 pb-24">
      <Suspense fallback={<p className="label text-muted">Loading the result…</p>}>
        <ResultPreview />
      </Suspense>
    </main>
  );
}