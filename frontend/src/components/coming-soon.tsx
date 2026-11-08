import Link from "next/link";

/** Placeholder for pages built in later steps, so the nav links already work. */
export function ComingSoon({ eyebrow, title, step }: { eyebrow: string; title: string; step: string }) {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 px-6 pt-36 pb-24">
      <p className="label text-brand-green">{eyebrow}</p>
      <h1 className="font-serif text-4xl text-brand-green italic">{title}</h1>
      <p className="text-muted">This page is built in step {step}.</p>
      <Link href="/" className="label w-fit text-brand-green-deep underline underline-offset-4">
        Back to home
      </Link>
    </main>
  );
}