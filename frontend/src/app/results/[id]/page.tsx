import type { Metadata } from "next";
import { Suspense } from "react";

import { ResultsView } from "@/components/results/results-view";

export const metadata: Metadata = { title: "Result" };

export default function ResultPage() {
  return (
    <main className="flex flex-1 flex-col">
      <Suspense fallback={<p className="label pt-40 text-center text-muted">Loading the result…</p>}>
        <ResultsView />
      </Suspense>
    </main>
  );
}