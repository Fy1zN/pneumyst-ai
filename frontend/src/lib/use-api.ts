"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback } from "react";

import { apiFetch } from "@/lib/api";

/** apiFetch with the signed-in user's Clerk session token attached; Clerk refreshes the token as needed. */
export function useApi() {
  const { getToken } = useAuth();
  return useCallback(
    async <T,>(path: string, init: RequestInit = {}): Promise<T> => apiFetch<T>(path, { ...init, token: await getToken() }),
    [getToken],
  );
}