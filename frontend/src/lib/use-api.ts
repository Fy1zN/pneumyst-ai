"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback } from "react";

import { apiFetch, apiFile } from "@/lib/api";

/** apiFetch with the signed-in user's Clerk session token attached; Clerk refreshes the token as needed. */
export function useApi() {
  const { getToken } = useAuth();
  return useCallback(
    async <T,>(path: string, init: RequestInit = {}): Promise<T> => apiFetch<T>(path, { ...init, token: await getToken() }),
    [getToken],
  );
}

/** Downloads a protected file. A plain link cannot send the sign-in token, so the file is fetched and then saved. */
export function useDownload() {
  const { getToken } = useAuth();
  return useCallback(
    async (path: string, fallbackName: string): Promise<void> => {
      const { blob, filename } = await apiFile(path, { token: await getToken() });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename ?? fallbackName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Give the browser a moment to start the download before the file's memory is freed.
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    },
    [getToken],
  );
}