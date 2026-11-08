# PNEUMYST AI frontend

Next.js 16 (App Router, TypeScript) and Tailwind CSS 4. The design it follows is in [`../docs/design/`](../docs/design/README.md).

## Run

Start the backend first (see the main [README](../README.md), step 11), then:

```powershell
Copy-Item .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
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
| `src/lib/api.ts` | The one place that calls the backend; errors become `ApiError` with the backend's `error_code` |
| `src/lib/api-types.ts` | Generated, do not edit by hand |

`NEXT_PUBLIC_` values are built into the website and visible to anyone, so never put a secret in `.env.local`.
