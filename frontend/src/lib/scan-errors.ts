import { ApiError } from "@/lib/api";

// Wording from docs/design/boards/error-states.png. The raw error code is never shown on its own.
const IMAGE_REASONS: Record<string, string> = {
  file_too_large: "This file is too big. The limit is 20 MB.",
  unsupported_format: "Only PNG and JPEG files work.",
  empty_file: "This file is empty.",
  unreadable: "This file could not be opened. It may be damaged.",
  too_small: "This image is too small to read. Use the original export, not a thumbnail.",
  too_large: "This image has too many pixels. Resize it and try again.",
  bad_aspect_ratio: "This image is too wide or too tall for a chest X-ray. Crop it to the chest.",
  blank_image: "This image looks blank.",
};

const FAILURES: Record<string, string> = {
  SERVER_BUSY: "The screening service is busy right now.",
  MODEL_UNAVAILABLE: "A screening model is not available right now.",
  INFERENCE_FAILED: "The models could not read this image.",
  INTERNAL_ERROR: "Something went wrong on our side.",
};

export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

export function imageMessage(reason: string): string {
  // The backend sends a reason such as "too_small" or "unsupported_format: only PNG and JPEG are accepted".
  const key = reason.split(":")[0].trim();
  return IMAGE_REASONS[key] ?? "This image could not be used. Try another PNG or JPEG.";
}

/** Checks done in the browser before uploading, so an obvious mistake does not cost a request. */
export function checkFile(file: File): string | null {
  const name = file.name.toLowerCase();
  const typeOk = ["image/png", "image/jpeg"].includes(file.type) || /\.(png|jpe?g)$/.test(name);
  if (!typeOk) return IMAGE_REASONS.unsupported_format;
  if (file.size === 0) return IMAGE_REASONS.empty_file;
  if (file.size > MAX_UPLOAD_BYTES) {
    return `This file is too big. It is ${(file.size / (1024 * 1024)).toFixed(1)} MB and the limit is 20 MB.`;
  }
  return null;
}

/** Where a failed /analyze call is shown and what it says. */
export type ScanProblem =
  | { kind: "image"; message: string }
  | { kind: "field"; field: string; message: string }
  | { kind: "limit"; minutes: number | null }
  | { kind: "failed"; message: string; reference: string | null; code: string }
  | { kind: "session" }
  | { kind: "offline" };

export function describe(error: unknown): ScanProblem {
  if (!(error instanceof ApiError)) return { kind: "failed", message: FAILURES.INTERNAL_ERROR, reference: null, code: "UNKNOWN" };
  if (error.code === "NETWORK_ERROR") return { kind: "offline" };
  if (error.status === 401 || error.code === "AUTH_UNAVAILABLE") return { kind: "session" };
  if (error.status === 413 || error.code === "FILE_TOO_LARGE") return { kind: "image", message: IMAGE_REASONS.file_too_large };
  if (error.code === "INVALID_IMAGE") return { kind: "image", message: imageMessage(error.message) };
  if (error.code === "INVALID_REQUEST") {
    // The backend names the field first, e.g. "case_number: String should match pattern ...".
    const field = error.message.split(":")[0].trim();
    const message = field === "case_number" ? "Use letters, digits, - or _ only, up to 32 characters." : "This answer was not accepted.";
    return { kind: "field", field, message };
  }
  if (error.status === 429) {
    return { kind: "limit", minutes: error.retryAfterSeconds ? Math.max(1, Math.ceil(error.retryAfterSeconds / 60)) : null };
  }
  return {
    kind: "failed",
    message: FAILURES[error.code] ?? FAILURES.INTERNAL_ERROR,
    reference: error.analysisId ? error.analysisId.slice(0, 8).toUpperCase() : null,
    code: error.code,
  };
}