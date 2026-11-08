import Link from "next/link";

import { Logo } from "@/components/logo";

const GITHUB_URL = "https://github.com/Fy1zN";
const LINKEDIN_URL = "https://www.linkedin.com/in/krish-malhotra-903092317/";

function GitHubIcon() {
  return (
    <svg viewBox="0 0 16 16" className="size-[18px]" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function LinkedInIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="currentColor" aria-hidden="true">
      <path d="M4.98 3.5C4.98 4.88 3.87 6 2.5 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM.22 8h4.56v14H.22V8zm7.5 0h4.37v1.92h.06c.61-1.15 2.1-2.36 4.32-2.36 4.62 0 5.47 3.04 5.47 7v7.44h-4.56v-6.6c0-1.57-.03-3.6-2.2-3.6-2.2 0-2.53 1.72-2.53 3.49v6.71H7.72V8z" />
    </svg>
  );
}

const textLink =
  "label border-b border-transparent pb-1 text-white/80 transition hover:border-brand-mint-strong hover:text-brand-mint-strong";
const iconLink =
  "grid size-[38px] place-items-center rounded-full border border-white/30 bg-white/10 text-white transition hover:bg-brand-mint-strong hover:text-brand-green-deep hover:shadow-[0_4px_10px_rgb(10_51_26_/_0.25)]";

export function SiteFooter() {
  return (
    <footer className="mt-auto bg-brand-green text-white">
      <div className="mx-auto flex max-w-6xl flex-col items-start gap-6 px-6 py-7 md:flex-row md:items-center md:justify-between">
        <Logo tone="light" />
        <nav aria-label="Footer" className="flex items-center gap-7">
          <Link href="/#privacy" className={textLink}>Privacy</Link>
          <Link href="/#how-it-works" className={textLink}>How it works</Link>
          <span aria-hidden="true" className="h-5 w-px bg-white/30" />
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" aria-label="GitHub" className={iconLink}>
            <GitHubIcon />
          </a>
          <a href={LINKEDIN_URL} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className={iconLink}>
            <LinkedInIcon />
          </a>
        </nav>
        <p className="text-xs text-white/75">© 2026 Krish Malhotra. Research prototype, not for clinical use.</p>
      </div>
    </footer>
  );
}