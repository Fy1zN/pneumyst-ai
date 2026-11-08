"use client";

import { useAuth, useClerk, useUser } from "@clerk/nextjs";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { AccountMenu, initialsOf } from "@/components/account-menu";
import { Logo } from "@/components/logo";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/scan", label: "New scan" },
  { href: "/history", label: "History" },
  { href: "/#how-it-works", label: "How it works" },
  { href: "/#privacy", label: "Privacy" },
];

function isActive(pathname: string, href: string): boolean {
  if (href.includes("#")) return false; // links to a section of the home page are never the current page
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

const bar =
  "flex w-full max-w-5xl items-center justify-between gap-6 rounded-full border border-white/30 bg-brand-green/85 py-2 pr-2 pl-5 shadow-[0_12px_32px_rgb(11_59_30_/_0.18)] backdrop-blur-xl";

/** Shown for the moment before the current page and sign-in state are known, so the bar never jumps. */
export function SiteNavFallback() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 flex justify-center px-4 pt-4">
      <div className={`${bar} min-h-14`}>
        <Logo tone="light" />
      </div>
    </header>
  );
}

const signInPill =
  "block rounded-full border border-white/40 bg-white/15 px-4 py-2 text-xs font-semibold tracking-[0.12em] text-white uppercase transition hover:bg-white/25";

export function SiteNav() {
  const pathname = usePathname();
  const { isLoaded, isSignedIn } = useAuth();
  const { user } = useUser();
  const { signOut } = useClerk();
  // The phone menu remembers which page it was opened on, so it closes by itself after navigating.
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === pathname;
  const onAuthPage = pathname.startsWith("/sign-in") || pathname.startsWith("/sign-up");
  const email = user?.primaryEmailAddress?.emailAddress ?? null;

  return (
    <header className="fixed inset-x-0 top-0 z-50 flex justify-center px-4 pt-4">
      <nav aria-label="Main" className={bar}>
        <Link href="/" aria-label="PNEUMYST AI home" className="rounded-full">
          <Logo tone="light" />
        </Link>

        <div className="hidden items-center gap-3 md:flex">
          <ul className="flex items-center gap-1">
            {LINKS.map((link) => {
              const active = isActive(pathname, link.href);
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    aria-current={active ? "page" : undefined}
                    className={
                      active
                        ? "glass-active block rounded-full border px-4 py-2 text-xs font-semibold tracking-[0.12em] text-brand-green-deep uppercase"
                        : "block rounded-full border border-transparent px-4 py-2 text-xs font-medium tracking-[0.12em] text-white/85 uppercase transition hover:border-white/40 hover:bg-white/15 hover:text-white"
                    }
                  >
                    {link.label}
                  </Link>
                </li>
              );
            })}
          </ul>
          {/* Keeps the bar the same width while Clerk is still loading. */}
          {!isLoaded ? (
            <span className="size-10" aria-hidden="true" />
          ) : isSignedIn ? (
            <AccountMenu />
          ) : (
            !onAuthPage && (
              <Link href="/sign-in" className={signInPill}>
                Sign in
              </Link>
            )
          )}
        </div>

        <button
          type="button"
          className="grid size-10 place-items-center rounded-full bg-white/15 text-white md:hidden"
          aria-expanded={open}
          aria-controls="phone-menu"
          aria-label={open ? "Close menu" : "Open menu"}
          onClick={() => setOpenOn(open ? null : pathname)}
        >
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </nav>

      {open && (
        <div id="phone-menu" className="absolute inset-x-4 top-20 rounded-3xl border border-white/70 bg-white/95 p-2.5 shadow-[0_20px_48px_rgb(11_59_30_/_0.22)] backdrop-blur-xl md:hidden">
          {isSignedIn && (
            <div className="mb-1 flex items-center gap-3 border-b border-line px-3 pt-2 pb-3">
              <span className="grid size-10 place-items-center rounded-full bg-brand-mint-strong text-sm font-semibold text-brand-green-deep">
                {initialsOf(user?.fullName, email)}
              </span>
              <span className="min-w-0">
                <span className="label block text-[0.5625rem] text-muted">Signed in as</span>
                <span className="block truncate text-sm font-semibold">{email}</span>
              </span>
            </div>
          )}
          <ul className="flex flex-col gap-1">
            {LINKS.map((link) => {
              const active = isActive(pathname, link.href);
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setOpenOn(null)}
                    className={
                      active
                        ? "glass-active flex items-center justify-between rounded-2xl border px-4 py-3 font-semibold text-brand-green-deep"
                        : "flex items-center justify-between rounded-2xl px-4 py-3 font-medium hover:bg-brand-mint"
                    }
                  >
                    {link.label}
                    <span aria-hidden="true" className="text-muted">›</span>
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="mt-1 border-t border-line px-2 pt-2">
            {isSignedIn ? (
              <button
                type="button"
                onClick={() => signOut({ redirectUrl: "/" })}
                className="w-full rounded-2xl px-2 py-3 text-left font-semibold text-band-likely"
              >
                Sign out
              </button>
            ) : (
              <Link href="/sign-in" onClick={() => setOpenOn(null)} className="block rounded-2xl px-2 py-3 font-semibold text-brand-green">
                Sign in
              </Link>
            )}
          </div>
        </div>
      )}
    </header>
  );
}