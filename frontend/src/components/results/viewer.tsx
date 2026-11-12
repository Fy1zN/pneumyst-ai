"use client";

import { useState } from "react";

import type { Analysis, Finding } from "@/lib/api";
import { BAND_DOT, when, type Band } from "@/lib/format";

type Mode = "side" | "overlay";

function noHeatmapReason(findings: Finding[]): string {
  if (findings.some((f) => f.heatmap === "not_requested")) return "Heatmaps were turned off for this scan.";
  if (!findings.some((f) => f.band === "likely" || f.band === "possible")) {
    return "Nothing was flagged as possible or likely, so no heatmaps were made.";
  }
  return "Heatmaps could not be made for this scan.";
}

/** A stored image; the links expire after a few minutes, so a failed load asks the page for fresh ones. */
function Picture({ src, alt, onExpired }: { src: string; alt: string; onExpired: () => void }) {
  // The images are the backend's signed links, which next/image would try to fetch and cache on the server.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} onError={onExpired} className="block size-full object-contain" />;
}

function Tag({ children }: { children: React.ReactNode }) {
  return <span className="label absolute top-2.5 left-2.5 rounded bg-black/55 px-2 py-1 text-[0.5625rem] text-white">{children}</span>;
}

/** Dark viewer: the X-ray the models saw and the chosen heatmap, side by side or laid over each other. */
export function Viewer({ analysis, onLinksExpired }: { analysis: Analysis; onLinksExpired: () => void }) {
  const links = analysis.images;
  const flaggedOrder = [...analysis.findings]
    .filter((f) => f.band === "likely" || f.band === "possible")
    .sort((a, b) => (a.band === b.band ? (b.score ?? 0) - (a.score ?? 0) : a.band === "likely" ? -1 : 1));
  const withMaps = flaggedOrder.filter((f) => links?.heatmaps[f.finding]);
  const [mode, setMode] = useState<Mode>("side");
  const [chosen, setChosen] = useState<string | null>(withMaps[0]?.finding ?? null);
  const [opacity, setOpacity] = useState(70);
  const current = withMaps.find((f) => f.finding === chosen) ?? withMaps[0] ?? null;
  const heatSrc = current && links ? links.heatmaps[current.finding] : null;
  const caseNumber = analysis.case_context?.case_number;
  const view = [analysis.case_context?.view, analysis.case_context?.position].filter(Boolean).join(", ");

  return (
    <section aria-label="X-ray viewer" className="flex flex-col gap-4 rounded-2xl border border-line bg-white p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label text-brand-green">X-ray viewer</p>
          <h1 className="mt-1 flex flex-wrap items-baseline gap-x-3 font-serif text-2xl text-brand-green italic">
            {caseNumber ?? "No case number"}
            <span className="font-sans text-sm text-muted not-italic">
              {when(analysis.created_at)}
              {view ? ` · ${view}` : ""}
            </span>
          </h1>
        </div>
        {heatSrc && (
          <div role="tablist" aria-label="View mode" className="flex rounded-lg border border-line bg-page p-0.5">
            {(["side", "overlay"] as const).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                onClick={() => setMode(m)}
                className={`label rounded-md px-3 py-1.5 text-[0.5625rem] ${mode === m ? "bg-white font-bold text-brand-green-deep shadow-sm" : "text-muted"}`}
              >
                {m === "side" ? "Side by side" : "Overlay"}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-xl bg-viewer p-4">
        {!links ? (
          <p className="py-24 text-center text-sm text-white/70">The images were not stored for this scan. The scores below are complete.</p>
        ) : heatSrc && current && mode === "side" ? (
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="relative aspect-square overflow-hidden rounded-lg bg-black">
              <Picture src={links.image} alt="Chest X-ray as analysed" onExpired={onLinksExpired} />
              <Tag>Original · 320 px</Tag>
            </div>
            <div className="relative aspect-square overflow-hidden rounded-lg bg-black">
              <Picture src={heatSrc} alt={`Heatmap for ${current.display_name}`} onExpired={onLinksExpired} />
              <Tag>Occlusion · {current.display_name}</Tag>
            </div>
          </div>
        ) : heatSrc && current ? (
          <div className="relative mx-auto aspect-square max-w-[30rem] overflow-hidden rounded-lg bg-black">
            <Picture src={links.image} alt="Chest X-ray as analysed" onExpired={onLinksExpired} />
            <div className="absolute inset-0" style={{ opacity: opacity / 100 }}>
              <Picture src={heatSrc} alt={`Heatmap for ${current.display_name}`} onExpired={onLinksExpired} />
            </div>
            <Tag>
              Overlay · {current.display_name} · {opacity}%
            </Tag>
          </div>
        ) : (
          <div className="mx-auto flex max-w-[30rem] flex-col gap-3">
            <div className="relative aspect-square overflow-hidden rounded-lg bg-black">
              <Picture src={links.image} alt="Chest X-ray as analysed" onExpired={onLinksExpired} />
              <Tag>Original · 320 px</Tag>
            </div>
            <p className="text-center text-sm text-white/70">{noHeatmapReason(analysis.findings)}</p>
          </div>
        )}
      </div>

      {heatSrc && current && (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Heatmap">
            <span className="label text-[0.5625rem] text-muted">Heatmap</span>
            {withMaps.map((f) => {
              const on = f.finding === current.finding;
              return (
                <button
                  key={f.finding}
                  type="button"
                  aria-pressed={on}
                  onClick={() => setChosen(f.finding)}
                  className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium ${on ? "border-brand-green-deep bg-brand-green-deep text-white" : "border-line bg-page text-ink"}`}
                >
                  <span className={`size-1.5 rounded-full ${BAND_DOT[f.band as Band]}`} aria-hidden="true" />
                  {f.display_name}
                </button>
              );
            })}
          </div>
          {mode === "overlay" && (
            <label className="flex min-w-48 flex-1 items-center gap-3">
              <span className="label text-[0.5625rem] text-muted">Opacity</span>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
                className="flex-1 accent-brand-green"
              />
              <span className="label w-9 text-right text-[0.625rem] font-bold text-brand-green-deep">{opacity}%</span>
            </label>
          )}
        </div>
      )}
      {heatSrc && (
        <p className="text-xs text-muted">
          Red marks the regions that changed the score most when hidden. A heatmap shows what influenced the model, not the outline of a lesion.
        </p>
      )}
    </section>
  );
}