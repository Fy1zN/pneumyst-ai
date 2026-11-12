"use client";

import { useState } from "react";

import type { Finding } from "@/lib/api";
import { BAND_DOT, BAND_TEXT, groupFindings, percent, type Band } from "@/lib/format";

const SHOW_UNLIKELY = 4;

function Row({ finding, band }: { finding: Finding; band: Band }) {
  const score = finding.score ?? 0;
  return (
    <li className="flex flex-col gap-1.5">
      <span className="flex items-baseline justify-between gap-3">
        <span className="truncate text-sm font-medium">{finding.display_name}</span>
        <span className="label text-[0.625rem] font-bold text-ink">{percent(score)}</span>
      </span>
      <span className="h-1 overflow-hidden rounded-full bg-line" aria-hidden="true">
        <span className={`block h-full rounded-full ${BAND_DOT[band]}`} style={{ width: `${Math.max(3, Math.min(100, score * 100))}%` }} />
      </span>
    </li>
  );
}

function Column({ band, items, children }: { band: Band; items: Finding[]; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl bg-page p-4">
      <p className={`label flex items-center gap-2 text-[0.625rem] font-bold ${BAND_TEXT[band]}`}>
        <span className={`size-2 rounded-full ${BAND_DOT[band]}`} aria-hidden="true" />
        {band} · {items.length}
      </p>
      {items.length ? (
        <ul className="flex flex-col gap-3">{children}</ul>
      ) : (
        <p className="text-sm text-muted">None</p>
      )}
    </div>
  );
}

/** All 15 findings: likely, possible and unlikely in columns; not banded and unavailable listed below, never hidden. */
export function FindingsPanel({ findings }: { findings: Finding[] }) {
  const groups = groupFindings(findings);
  const [allUnlikely, setAllUnlikely] = useState(false);
  const unlikely = allUnlikely ? groups.unlikely : groups.unlikely.slice(0, SHOW_UNLIKELY);

  return (
    <section aria-labelledby="findings-title" className="flex flex-col gap-4 rounded-2xl border border-line bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="findings-title" className="label text-brand-green">
          Findings by band
        </h2>
        <p className="text-xs text-muted">{findings.length} findings from 2 models</p>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        <Column band="likely" items={groups.likely}>
          {groups.likely.map((f) => (
            <Row key={f.finding} finding={f} band="likely" />
          ))}
        </Column>
        <Column band="possible" items={groups.possible}>
          {groups.possible.map((f) => (
            <Row key={f.finding} finding={f} band="possible" />
          ))}
        </Column>
        <Column band="unlikely" items={groups.unlikely}>
          {unlikely.map((f) => (
            <Row key={f.finding} finding={f} band="unlikely" />
          ))}
          {groups.unlikely.length > SHOW_UNLIKELY && (
            <li>
              <button type="button" onClick={() => setAllUnlikely((v) => !v)} className="text-sm font-semibold text-brand-green">
                {allUnlikely ? "Show fewer" : `Show all ${groups.unlikely.length}`}
              </button>
            </li>
          )}
        </Column>
      </div>
      {(groups.notBanded.length > 0 || groups.unavailable.length > 0) && (
        <div className="flex flex-col gap-1 border-t border-line pt-3 text-sm">
          {groups.notBanded.map((f) => (
            <p key={f.finding}>
              <span className="font-medium">{f.display_name}</span>
              <span className="text-muted">
                {" "}
                · {f.score === null ? "no score" : percent(f.score)} · not banded: too few validation cases for a reliable band, so the score is
                shown but not interpreted.
              </span>
            </p>
          ))}
          {groups.unavailable.map((f) => (
            <p key={f.finding}>
              <span className="font-medium">{f.display_name}</span>
              <span className="text-band-likely"> · unavailable: this model did not run, so there is no score.</span>
            </p>
          ))}
        </div>
      )}
      <p className="text-xs text-muted">Each band compares the calibrated score with thresholds fixed on validation data.</p>
    </section>
  );
}