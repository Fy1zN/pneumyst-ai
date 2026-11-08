import type { Metadata } from "next";

import { ComingSoon } from "@/components/coming-soon";

export const metadata: Metadata = { title: "History" };

export default function HistoryPage() {
  return <ComingSoon eyebrow="Your scans" title="Scan history" step="F5" />;
}