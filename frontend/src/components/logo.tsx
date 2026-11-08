type Tone = "light" | "green";

export function LungMark({ className, tone = "green" }: { className?: string; tone?: Tone }) {
  const colour = tone === "light" ? "#ffffff" : "#16a34a";
  return (
    <svg viewBox="0 0 40 40" fill="none" aria-hidden="true" className={className}>
      <circle cx="20" cy="20" r="18" stroke={colour} strokeWidth="2.5" />
      <path d="M20 9v10" stroke={colour} strokeWidth="2.5" strokeLinecap="round" />
      <path
        d="M18.6 18c-2 .6-3.4 2.6-4.6 5.2-1.2 2.7-1.8 5.4-1.6 6.6.2 1.1 1.3 1.4 2.8.8 2.4-1 3.8-3 4-6.4V18.6Z"
        fill={colour}
      />
      <path
        d="M21.4 18c2 .6 3.4 2.6 4.6 5.2 1.2 2.7 1.8 5.4 1.6 6.6-.2 1.1-1.3 1.4-2.8.8-2.4-1-3.8-3-4-6.4V18.6Z"
        fill={colour}
      />
    </svg>
  );
}

export function Logo({ tone = "green" }: { tone?: Tone }) {
  const light = tone === "light";
  return (
    <span className="flex items-center gap-2.5">
      <LungMark tone={tone} className="size-9 shrink-0" />
      <span className="flex flex-col leading-none">
        <span className={`font-logo text-lg font-extrabold tracking-tight ${light ? "text-white" : "text-brand-green-deep"}`}>
          PNEUMYST
        </span>
        <span className={`font-logo text-[0.6rem] font-medium tracking-[0.22em] ${light ? "text-brand-mint-strong" : "text-brand-green"}`}>
          AI SCREENING
        </span>
      </span>
    </span>
  );
}