import type { Metadata } from "next";
import Image from "next/image";
import { Suspense } from "react";

import { ScanForm } from "@/components/scan/scan-form";

export const metadata: Metadata = { title: "New scan" };

export default function ScanPage() {
  return (
    <main className="relative flex-1 overflow-hidden">
      {/* The lungs sit behind the frosted panels, as in docs/design/screens/new-scan.png. */}
      <Image
        src="/lung-artwork.png"
        alt=""
        width={1536}
        height={1024}
        priority
        className="pointer-events-none absolute top-24 left-1/2 w-[70rem] max-w-none -translate-x-1/2 opacity-70 mix-blend-multiply select-none"
      />
      {/* The form reads the signed-in user, which is only known at request time. */}
      <Suspense fallback={<div className="min-h-[60rem]" />}>
        <ScanForm />
      </Suspense>
    </main>
  );
}