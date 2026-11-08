import { Suspense, type ReactNode } from "react";

/** Centres Clerk's sign-in or register card on the mint page, below the glass nav. */
export function AuthCardPage({ children }: { children: ReactNode }) {
  return (
    <main className="flex flex-1 flex-col items-center gap-5 px-4 pt-32 pb-20">
      {/* Clerk's card reads the URL, which is only known at request time; this keeps the space while it loads. */}
      <Suspense fallback={<div className="h-[30rem] w-full max-w-[25rem] rounded-[18px] border border-line bg-white" />}>
        {children}
      </Suspense>
      <p className="text-xs text-muted">Research prototype. Not for clinical use.</p>
    </main>
  );
}