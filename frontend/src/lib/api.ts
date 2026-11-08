import type { components } from "@/lib/api-types";

export type Health = components["schemas"]["HealthResponse"];
export type AnalysisPage = components["schemas"]["AnalysisPage"];
type ErrorBody = components["schemas"]["ErrorResponse"];

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Every failed call becomes one of these, with the backend's error_code so pages can pick the right message. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly analysisId: string | null = null,
    readonly retryAfterSeconds: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Options = RequestInit & { token?: string | null };

export async function apiFetch<T>(path: string, { token, headers, ...init }: Options = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}), ...headers },
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError(0, "NETWORK_ERROR", "Cannot reach PNEUMYST. Check your connection.");
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as Partial<ErrorBody> | null;
    const retry = Number(response.headers.get("Retry-After"));
    throw new ApiError(
      response.status,
      body?.error_code ?? "HTTP_ERROR",
      body?.detail ?? response.statusText,
      body?.analysis_id ?? null,
      Number.isFinite(retry) && retry > 0 ? retry : null,
    );
  }
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return apiFetch<Health>("/health", { signal, cache: "no-store" });
}