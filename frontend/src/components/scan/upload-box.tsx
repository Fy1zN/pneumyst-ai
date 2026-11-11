"use client";

import { useEffect, useMemo, useRef, useState } from "react";

function size(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Click or drag an X-ray in; shows a preview once chosen and a red message when the file cannot be used. */
export function UploadBox({
  file,
  error,
  disabled,
  onFile,
}: {
  file: File | null;
  error: string | null;
  disabled?: boolean;
  onFile: (file: File | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  // Free the preview's memory when the file changes or the page closes.
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const pick = () => {
    if (!disabled) input.current?.click();
  };

  const tone = error
    ? "border-band-likely bg-[#fdecea]"
    : dragging
      ? "border-brand-green bg-brand-mint"
      : "border-brand-green/60 bg-brand-mint/40 hover:bg-brand-mint/70";

  return (
    <div className="flex flex-1 flex-col">
      <input
        ref={input}
        type="file"
        accept="image/png,image/jpeg"
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          onFile(event.target.files?.[0] ?? null);
          event.target.value = ""; // choosing the same file again still triggers a change
        }}
      />
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label={file ? "Change the chest X-ray" : "Choose a chest X-ray"}
        aria-describedby={error ? "upload-error" : undefined}
        onClick={pick}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            pick();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) onFile(event.dataTransfer.files?.[0] ?? null);
        }}
        className={`flex min-h-72 flex-1 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-6 text-center transition ${tone} ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
      >
        {error ? (
          <>
            <span className="grid size-11 place-items-center rounded-full bg-white text-xl font-bold text-band-likely" aria-hidden="true">
              !
            </span>
            <p id="upload-error" className="max-w-xs font-semibold text-band-likely" role="alert">
              {error}
            </p>
            <span className="label rounded-lg border border-line bg-white px-4 py-2.5 font-bold text-brand-green-deep">
              Choose another file
            </span>
          </>
        ) : file && preview ? (
          <>
            {/* A local preview from the user's own file; next/image cannot optimise a blob URL. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="The chosen chest X-ray" className="max-h-56 rounded-lg bg-viewer object-contain" />
            <p className="max-w-full truncate text-sm font-semibold text-brand-green-deep">{file.name}</p>
            <p className="text-xs text-muted">
              {size(file.size)} · <span className="font-semibold text-brand-green underline underline-offset-4">Change</span>
            </p>
          </>
        ) : (
          <>
            <span className="grid size-14 place-items-center rounded-full border border-brand-mint-strong bg-white text-2xl text-brand-green" aria-hidden="true">
              +
            </span>
            <p className="font-medium text-brand-green-deep">Click to upload a chest X-ray</p>
            <p className="text-xs text-muted">or drag it here · PNG or JPEG, up to 20 MB</p>
          </>
        )}
      </div>
    </div>
  );
}