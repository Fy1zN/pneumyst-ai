# PNEUMYST AI frontend

Next.js 16 (App Router, TypeScript) and Tailwind CSS 4. The design it follows is in [`../docs/design/`](../docs/design/README.md).

## Run

Start the backend first (see the main [README](../README.md), step 11), then:

```powershell
Copy-Item .env.example .env.local   # then paste the two Clerk keys from dashboard.clerk.com > API keys
npm install
npm run dev
```

Open `http://localhost:3000`.

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Development server |
| `npm run api:types` | Regenerates `src/lib/api-types.ts` from the running backend's `/openapi.json` |
| `npm run typecheck` | TypeScript check |
| `npm run lint` | ESLint |
| `npm run build` | Production build |

## Layout

| Path | Contents |
|---|---|
| `src/app/globals.css` | Colour and font tokens from the design, the `label` and `glass-active` styles |
| `src/app/layout.tsx` | Fonts, glass nav and footer around every page |
| `src/components/` | Nav, footer, logo and shared parts |
| `src/proxy.ts` | Clerk check before every page; `/scan`, `/history` and `/results` need sign-in |
| `src/lib/api.ts` | The one place that calls the backend; errors become `ApiError` with the backend's `error_code` |
| `src/lib/use-api.ts` | `useApi()`: `apiFetch` with the signed-in user's Clerk token |
| `src/lib/clerk-theme.ts` | Styling and wording for Clerk's sign-in and register cards |
| `src/components/scan/` | New scan page: case form, upload box, progress and error cards |
| `src/lib/case-fields.ts` | The 19 case fields and their choices, typed against the API |
| `src/lib/scan-errors.ts` | Backend error codes turned into the wording users see |
| `src/lib/api-types.ts` | Generated, do not edit by hand |

`NEXT_PUBLIC_` values are built into the website and visible to anyone. `CLERK_SECRET_KEY` stays server-side; `.env.local` is git-ignored and must never be committed.
