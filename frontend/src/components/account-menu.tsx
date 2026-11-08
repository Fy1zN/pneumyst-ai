"use client";

import { useClerk, useUser } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

export function initialsOf(name: string | null | undefined, email: string | null | undefined): string {
  const source = (name ?? "").trim() || (email ?? "").split("@")[0];
  const parts = source.split(/[\s._-]+/).filter(Boolean);
  const letters = parts.length > 1 ? parts[0][0] + parts[1][0] : source.slice(0, 2);
  return letters.toUpperCase() || "?";
}

const ITEMS = [
  { href: "/history", label: "Your scans" },
  { href: "/scan", label: "New scan" },
  { href: "/#privacy", label: "Privacy" },
];

/** The avatar in the glass nav and the menu it opens (docs/design/screens/history-account-menu.png). */
export function AccountMenu() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  // Close on a click outside the menu or on Escape.
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (box.current && !box.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) return null;
  const email = user.primaryEmailAddress?.emailAddress ?? null;

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account menu"
        onClick={() => setOpen((v) => !v)}
        className={`grid size-10 place-items-center rounded-full bg-brand-mint-strong text-sm font-semibold text-brand-green-deep transition ${open ? "ring-2 ring-white" : "hover:ring-2 hover:ring-white/60"}`}
      >
        {initialsOf(user.fullName, email)}
      </button>

      {open && (
        <div
          role="menu"
          className="absolute top-14 right-0 w-64 rounded-2xl border border-line bg-white p-1.5 text-ink shadow-[0_16px_40px_rgb(11_59_30_/_0.16)]"
        >
          <div className="border-b border-line px-3 pt-2.5 pb-3">
            <p className="label text-[0.5625rem] text-muted">Signed in as</p>
            <p className="truncate text-sm font-semibold">{email}</p>
          </div>
          {ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              role="menuitem"
              onClick={() => setOpen(false)}
              className="block rounded-lg px-3 py-2.5 text-sm font-medium hover:bg-brand-mint"
            >
              {item.label}
            </Link>
          ))}
          <div className="my-1 h-px bg-line" />
          <button
            type="button"
            role="menuitem"
            onClick={() => signOut({ redirectUrl: "/" })}
            className="block w-full rounded-lg px-3 py-2.5 text-left text-sm font-medium text-band-likely hover:bg-[#fdecea]"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}