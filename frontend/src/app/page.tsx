import Image from "next/image";
import Link from "next/link";

import { HealthBadge } from "@/components/health-badge";

// Temporary home page for F1: the hero from the design plus a backend check. The full landing page comes in F5.
export default function Home() {
  return (
    <main className="relative overflow-hidden">
      <section className="mx-auto grid max-w-6xl items-center gap-10 px-6 pt-36 pb-24 md:grid-cols-[1.2fr_1fr]">
        <div className="relative z-10 flex flex-col gap-6">
          <p className="label text-brand-green">Explainable chest X-ray screening</p>
          <h1 className="font-serif text-5xl leading-[1.08] text-brand-green italic md:text-[3.5rem]">
            Read the X-ray.
            <br />
            See why it was flagged.
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-muted">
            PNEUMYST screens a chest X-ray for tuberculosis and 14 other findings, shows which parts of the image drove
            each result, and writes a short summary you can check against it.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/scan"
              className="label rounded-lg bg-brand-green px-7 py-4 font-bold text-white transition hover:bg-brand-green-deep"
            >
              Start a new scan
            </Link>
            <Link
              href="/#how-it-works"
              className="label rounded-lg border border-brand-green px-7 py-4 font-bold text-brand-green-deep transition hover:bg-brand-mint"
            >
              How it works
            </Link>
          </div>
          <p className="text-sm text-muted">Research prototype. Not a diagnosis and not for clinical use.</p>
          <HealthBadge />
        </div>
        <Image
          src="/lung-artwork.png"
          alt=""
          width={1536}
          height={1024}
          priority
          className="pointer-events-none w-full mask-[radial-gradient(ellipse_at_center,black_50%,transparent_72%)] mix-blend-multiply select-none md:scale-125"
        />
      </section>
    </main>
  );
}