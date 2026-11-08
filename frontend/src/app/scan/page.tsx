import type { Metadata } from "next";

import { ComingSoon } from "@/components/coming-soon";

export const metadata: Metadata = { title: "New scan" };

export default function ScanPage() {
  return <ComingSoon eyebrow="New scan" title="Upload a chest X-ray" step="F3" />;
}